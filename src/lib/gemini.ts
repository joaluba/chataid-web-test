import { GoogleGenAI, Modality } from "@google/genai";

export const INSTRUCTION_PROMPT = `
You are a research assistant for a study about communication in different noise settings. 
Your goal is to explain the study to the participant and ensure they understand what to do.

INSTRUCTIONS:
1. Greet the participant warmly and read the exact message: "Hi there! Welcome! You are here because you agreed to take part in the study about communication 
in different noise settings. Together, we will simulate a common communication scenario you can find in real life and your task will be to have a natural conversation with me. 
As is often the case in the normal life, you will have to collect specific information from me and note them down. 
Now take a moment to scroll down to see the user interface. On the left you can see a list of all the information you
need to collect and on the right there are empty fields for you to fill in. Let me know when you are ready to continue."
2. Wait until the participant confirms they are ready.
3. Continue the instruction by reading the exact message: "Ok, let's start with the first scenario! The first scenario is a conversation with a waiter in a cafe. 
I will be playing the role of your waiter. You have to collect the information that you can see displayed on the screen now. 
If you haven't understood, you can always ask for a repetition or clarification.  Are you ready?"
3. When they say they are ready or have no more questions, say: 
"Great! Please press 'STOP INSTRUCTION' and then 'START EXPERIMENT' to begin the cafe scenario."
4. Keep responses brief and helpful.

PARTICIPANT'S TASK:
Have a role-play natural conversation with the agent and collect requested information. 
Whenever unsure, ask for a repetition. Information to collect:
- Price of coffee with milk
- Available milk options
- Is there extra charge for vegan milk
- Specialty cake
- Wifi Name
- Wifi Password
- Maximum table duration
- Today's event
- Name of the artist
- Café closing time
`;

export const EXPERIMENT_PROMPT = `
You are Ramona, a friendly and welcoming waiter at a cozy café "Coffee and Jazz" in Barcelona. 
You are helpful, empathetic, and speak naturally—like in a real conversation.

CONTEXT & FIXED INFORMATION:
- Price of coffee with milk: 2.5 euros
- Available milk options: Almond, Coconut, Cow Milk
- Is there extra charge for vegan milk: Yes, 20 cents.
- Specialty cake: Our famous Tarta de Santiago (almond cake)
- Wifi Name: CoffeeAndJazz written together
- Wifi Password: EnjoyYourCoffee written together
- Maximum table duration: 90 minutes
- Today's event: Jazz concert
- Name of the artist: Barcelona Jazz Collective
- Café closing time: 10pm


GOODBYE INSTRUCTION (When session ends):
When a person asks to close a session, ask once if they are sure. 
If they say they are sure 
"Thanks for your participation! Please press STOP to close the session. Have a great day!"

CORE RULES:
1. Keep responses to maximum 2 sentences (natural for voice interaction)
2. Information from "CONTEXT & FIXED INFORMATION" section are secret. The user has to ask for them specifically. 
NEVER voluntarily share any information from CONTEXT & FIXED INFORMATION. 
3. Stay in character at all times; respond as a real waiter would
4. If asked about something you don't know, say so naturally ("I'm not sure about that")
5. Use conversational language (contractions, natural phrases)
6. If the user asks you to repeat something, do it. 

CONVERSATION GUIDELINES:
- Ask clarifying questions if a request is vague
- Be naturally conversational (no robotic responses)
- Show personality—make small talk feel genuine
- If someone asks multiple questions at once, answer one at a time (more natural)
`;

// --- SIGNAL FLOW GRAPH SETUP ---
/* 
  * --- SIGNAL FLOW: AGENT VOICE ---
  * [AudioBufferSource] (Gemini Speech)
  *    |
  *    +--> [analyser] (Visualizer Probe)
  *    |
  *    v
  * [voiceGainNode] (Unity Gain Primary Input)
  *    |
  *    +----(Dry Path)-----> [dryGainNode] (gain mute if phase=2) ----> [Destination]
  *    |
  *    +----(Wet Path)-----> [convolver] (HRTF)
  *    |                        |
  *    |                        v
  *    |                 [speechRMSProcessor]
  *    |                        |
  *    |                        +-----> [Destination] (Binaural) (samples mute if phase=1)
  *    |                        |
  *    |                        v
  *    |                 [voiceRecordingProcessor] ----> [Destination] (samples mute always)
  *    |                        (Saves Stereo Samples)
  *    |
  *    +----(Recording)----> [transcriptProcessor] --------------> [Destination] (samples mute always)
  *                               (Saves Mono Mixed Samples)
  *
  * --- SIGNAL FLOW: BACKGROUND NOISE ---
  * [AudioBufferSource] (Noise File)
  *    |
  *    v
  * [noiseGainNode] (Gain adjusted for SNR) (gain mute if phase=1)
  *    |
  *    +----(Playback)------------------------------------------> [Destination]
  *    |
  *    +----(Recording)----> [noiseRecordingProcessor] ----------> [Destination] (samples mute always)
  *                               (Saves Stereo Samples)
  *
  * --- SIGNAL FLOW: USER MICROPHONE ---
  * [MediaStreamSource] (Microphone)
  *    |
  *    +----(Transmission)--> [processor] (PCM -> Gemini) ------> [Destination] (samples mute always)
  *    |
  *    +----(Recording)-----> [transcriptProcessor] ------------> [Destination] (samples mute always)
  *                               (Saves Mono Mixed Samples)
  */

export class LiveAudioSession {

  // ----- PARAMETERS OF ACOUSTIC ENVIRONMENT ----
  public static readonly SNR_DB = 5; 
  private static readonly SIGNAL_VOLUME = 0.8;
  private static readonly NOISE_FILE_URL = "https://res.cloudinary.com/dqttqwfib/video/upload/v1776700340/cafe_noise_bin_pbsmfe.mp3";
  private static readonly CUSTOM_HRIR_URL = "https://res.cloudinary.com/dqttqwfib/video/upload/v1776699826/cafe_rir_bin_pnaczh.wav"; 

  // ----- GEMINI VOICE API CORE -----
  /** Instance of the Google Generative AI client */
  private ai: any;
  /** Active Gemini Multimodal Live session instance */
  private session: any;
  /** Shared promise used to coordinate async session initialization */
  private sessionPromise: Promise<any> | null = null;

  // ----- WEB AUDIO INFRASTRUCTURE -----
  /** Primary controller for the Web Audio processing graph (The "Brain") */
  private audioContext: AudioContext | null = null;   
  /** Stores the decoded audio data of the HRIR/Convolution file */
  private hrirBuffer: AudioBuffer | null = null;
  /** Custom audio destination, used if routing to an external stream is required */
  private recordingDestination: MediaStreamAudioDestinationNode | null = null;

  // ----- USER INPUT (MICROPHONE) -----
  /** Captures local microphone input for transmission to Gemini */
  private processor: ScriptProcessorNode | null = null;
  /** Web Audio source node wrapping the user's microphone stream */
  private source: MediaStreamAudioSourceNode | null = null;
  /** Reference to the local microphone MediaStream */
  private stream: MediaStream | null = null;

  // ----- AGENT PLAYBACK (FROM GEMINI) -----
  /** Queue of PCM audio chunks received from Gemini, awaiting playback */
  private audioQueue: Int16Array[] = [];
  /** Indicates if the agent is currently playing back audio */
  private isPlaying = false;
  /** The precise AudioContext time at which the next audio chunk should begin playback */
  private nextStartTime = 0;

  // ----- SIGNAL PROCESSING CHAIN -----
  /** Applies spatial/Room Impulse Response (RIR) effects to the speech signal */
  private convolver: ConvolverNode | null = null;
  /** Controls volume of the spatialized (processed) speech */
  private voiceGainNode: GainNode | null = null;
  /** Controls volume of the unprocessed (dry) speech signal */
  private dryGainNode: GainNode | null = null;
  /** Node for capturing frequency/time-domain data for visualizer meters */
  private analyser: AnalyserNode | null = null;
  /** Source node responsible for playing the background noise track */
  private noiseSource: AudioBufferSourceNode | null = null;

  // ----- RECORDING & WAV EXPORT LOGIC -----
  /** Processor node dedicated to measuring the real-time energy (RMS) of speech */
  private speechRMSProcessor: ScriptProcessorNode | null = null;
  /** Processor node capturing the mix for transcript validation */
  private transcriptProcessor: ScriptProcessorNode | null = null;
  /** Processor node capturing binaural speech for export to WAV */
  private voiceRecordingProcessor: ScriptProcessorNode | null = null;
  /** Processor node capturing the background noise component for export to WAV */
  private noiseRecordingProcessor: ScriptProcessorNode | null = null;

  // ----- ACCUMULATED AUDIO SAMPLES (FOR SAVING) -----
  /** Accumulates raw samples for the transcript-level (mix) recording */
  private transcriptSamples: Float32Array[] = [];
  /** Left channel samples for the clean binaural speech recording */
  private voiceSamplesL: Float32Array[] = [];
  /** Right channel samples for the clean binaural speech recording */
  private voiceSamplesR: Float32Array[] = [];
  /** Left channel samples for the background noise component recording */
  private noiseSamplesL: Float32Array[] = [];
  /** Right channel samples for the background noise component recording */
  private noiseSamplesR: Float32Array[] = [];

  // ----- MEASUREMENT STATISTICS & SNR LOGIC -----
  /** Accumulates the sum of squared amplitudes for energy calculation */
  private speechSumSquares = 0;
  /** Tracks total samples processed to determine average energy per sample */
  private speechSampleCount = 0;
  /** Sum of squares for input signal before any gain/processing */
  private preGraphSpeechSumSquares = 0;
  /** Sample count for pre-processing energy measurement */
  private preGraphSpeechSampleCount = 0;
  /** Final Root-Mean-Square level of the speech signal used to calibrate noise volume */
  private measuredSpeechRMS = 0.1; // Default fallback

  constructor(apiKey: string) {
    if (!apiKey) {
      throw new Error("Gemini API key is required to start a session.");
    }
    this.ai = new GoogleGenAI({ apiKey });
  }

 // --------------- HELPER: LOAD AUDIO FROM FILE ------------
  private async loadAudioFromFile(context: AudioContext, url: string): Promise<AudioBuffer> {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    return await context.decodeAudioData(arrayBuffer);
  }


 // --------------- HELPER: COMPUTE RMS  ------------
   private computeRMS(buffer: AudioBuffer): number {
    let sumSquares = 0;
    const length = buffer.length;
    const numChannels = buffer.numberOfChannels;

    for (let i = 0; i < length; i++) {
      let sampleSum = 0;
      for (let ch = 0; ch < numChannels; ch++) {
        sampleSum += buffer.getChannelData(ch)[i];
      }
      const averagedSample = sampleSum / numChannels;
      sumSquares += averagedSample * averagedSample;
    }
    return length > 0 ? Math.sqrt(sumSquares / length) : 0;
  }


 // --------------- FUNCTION TO START THE NOISE SOURCE ------------
 // takes the measured rms of speech at the input and sets noise level according to a chosen SNR
  private async startNoise(speechRMS: number, audible: boolean) {
    if (!this.audioContext) return; 
    
    let buffer: AudioBuffer;
    
    try {
      buffer = await this.loadAudioFromFile(this.audioContext, LiveAudioSession.NOISE_FILE_URL);
    } catch (err) {
      console.error("Failed to load noise buffer:", err);
      return; 
    }

    this.noiseSource = this.audioContext.createBufferSource();
    this.noiseSource.buffer = buffer;
    this.noiseSource.loop = true;

    const noiseRMS = this.computeRMS(buffer);
    const targetGain = speechRMS / (noiseRMS * Math.pow(10, LiveAudioSession.SNR_DB / 20));
    
    const noiseGainNode = this.audioContext.createGain();
    // gain mute if phase=1 (audible=false), otherwise use calibrated SNR gain
    noiseGainNode.gain.value = audible ? targetGain : 0; 

    // Record noise (stereo)
    this.noiseRecordingProcessor = this.audioContext.createScriptProcessor(4096, 2, 2);
    this.noiseRecordingProcessor.onaudioprocess = (e) => {
      const left = e.inputBuffer.getChannelData(0);
      const right = e.inputBuffer.getChannelData(1);
      this.noiseSamplesL.push(new Float32Array(left));
      this.noiseSamplesR.push(new Float32Array(right));
      
      e.outputBuffer.getChannelData(0).fill(0); // samples mute
      e.outputBuffer.getChannelData(1).fill(0); // samples mute
    };

    // Connect noise source to audio output for the user
    this.noiseSource.connect(noiseGainNode);
    noiseGainNode.connect(this.audioContext.destination);

    /** This is a "keep-alive" connection: ScriptProcessorNode often stops working 
     * if it isn't connected to an active output. However, since the code inside
     * the processor uses .fill(0) to silence its output, the line below doesnt produce 
     * any sound.*/ 
    // file -> noise source -> noiseGainNode (gain mute if phase=1) -> playback
    //                                         |
    //                                         +-> recording -> destination (samples mute)

    noiseGainNode.connect(this.noiseRecordingProcessor);
    this.noiseRecordingProcessor.connect(this.audioContext.destination);

    this.noiseSource.start();

  }


 // ---------------- MAIN ENTRY POINT FOR AI AUDIO SESSION ------------
  async start(params: {
    systemInstruction: string; // prompt
    shouldPlayNoise: boolean; // no noise (phase 1) or yes noise (phase 2)
    onError?: (error: any) => void;
    onClose?: () => void;
  }) {
    const { systemInstruction, shouldPlayNoise, onError, onClose } = params;

    // request access to users microphone
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.audioContext = new AudioContext({ sampleRate: 24000 });
      
      if (this.audioContext.state === "suspended") {
        await this.audioContext.resume();
      }

    // RESETTING RECORDING ARRAYS
    if (!shouldPlayNoise) {
      // Starting Phase 1: Reset everything
      this.speechSumSquares = 0;
      this.speechSampleCount = 0;
      this.preGraphSpeechSumSquares = 0;
      this.preGraphSpeechSampleCount = 0;
      
      this.transcriptSamples = [];
      this.voiceSamplesL = [];
      this.voiceSamplesR = [];
      this.noiseSamplesL = [];
      this.noiseSamplesR = [];
    } else {
      // Starting Phase 2: Clear binaural and noise samples to ensure they are synchronized for this phase
      this.voiceSamplesL = [];
      this.voiceSamplesR = [];
      this.noiseSamplesL = [];
      this.noiseSamplesR = [];
    }
      
      // 1. Primary Gain Node (Unity gain to avoid clipping)
      this.voiceGainNode = this.audioContext.createGain();
      this.voiceGainNode.gain.value = LiveAudioSession.SIGNAL_VOLUME;

      // 2. Dry Path Setup (Passthrough for Phase 1 or debugging)
      this.dryGainNode = this.audioContext.createGain();
      this.dryGainNode.gain.value = shouldPlayNoise ? 0 : 1; // gain mute if phase=2
      this.voiceGainNode.connect(this.dryGainNode);
      this.dryGainNode.connect(this.audioContext.destination);

      // 3. Wet Path (Spatial/HRTF processing via Convolver)
      this.convolver = this.audioContext.createConvolver();
      if (!this.hrirBuffer) {
        this.hrirBuffer = await this.loadAudioFromFile(this.audioContext, LiveAudioSession.CUSTOM_HRIR_URL);
      }
      this.convolver.buffer = this.hrirBuffer;
      this.voiceGainNode.connect(this.convolver);

      // 4. RMS Calculation & Wet Playback Control
      this.speechRMSProcessor = this.audioContext.createScriptProcessor(4096, 2, 2);
      this.speechRMSProcessor.onaudioprocess = (e) => {
        let sumSq = 0;
        const length = e.inputBuffer.length;
        const numChannels = e.inputBuffer.numberOfChannels;

        // Measurement and Muting Logic
        if (!shouldPlayNoise) {
          // samples mute if phase=1
          for (let ch = 0; ch < numChannels; ch++) {
            e.outputBuffer.getChannelData(ch).fill(0);
          }

          for (let i = 0; i < length; i++) {
            let sampleSum = 0;
            for (let ch = 0; ch < numChannels; ch++) {
              sampleSum += e.inputBuffer.getChannelData(ch)[i];
            }
            const averagedSample = sampleSum / numChannels;
            sumSq += averagedSample * averagedSample; // sum of squares (energy for 1 block of data)
          }

          this.speechSumSquares += sumSq;
          this.speechSampleCount += length;
          
        } else {
          // In Phase 2, just passthrough the wet audio
          for (let ch = 0; ch < numChannels; ch++) {
            const input = e.inputBuffer.getChannelData(ch);
            const output = e.outputBuffer.getChannelData(ch);
            output.set(input);
          }
        }
      };

      this.convolver.connect(this.speechRMSProcessor);
      this.speechRMSProcessor.connect(this.audioContext.destination);

      // 5. Recording Processors (WAV Export)

      // Transcript (Mono Agent Voice + Mono User Mic)
      this.transcriptProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);
      this.transcriptProcessor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        this.transcriptSamples.push(new Float32Array(input));
        e.outputBuffer.getChannelData(0).fill(0); //mute audio at the output
      };
      this.voiceGainNode.connect(this.transcriptProcessor);
      this.transcriptProcessor.connect(this.audioContext.destination);

      // Voice (Stereo Binaural post-HRTF)
      this.voiceRecordingProcessor = this.audioContext.createScriptProcessor(4096, 2, 2);
      this.voiceRecordingProcessor.onaudioprocess = (e) => {
        const left = e.inputBuffer.getChannelData(0);
        const right = e.inputBuffer.getChannelData(1);
        this.voiceSamplesL.push(new Float32Array(left));
        this.voiceSamplesR.push(new Float32Array(right));
        e.outputBuffer.getChannelData(0).fill(0);
        e.outputBuffer.getChannelData(1).fill(0);
      };
      // because speech RMS processor is muted in phase 1, the voice recording will be empty in phase 1
      this.speechRMSProcessor.connect(this.voiceRecordingProcessor);

      this.voiceRecordingProcessor.connect(this.audioContext.destination);
 
      // 6. Visualizer Probe
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;
 
      // 7. Noise Setup (Initialize in both phases for synchronization)
      // In Phase 1: noise is muted via gain ("gain mute if phase=1")
      // In Phase 2: noise gain is calculated based on measured speaker RMS
      if (shouldPlayNoise && this.speechSampleCount > 0) {
        this.measuredSpeechRMS = Math.sqrt(this.speechSumSquares / this.speechSampleCount);
      }
      this.startNoise(this.measuredSpeechRMS, shouldPlayNoise);
 
      this.sessionPromise = this.ai.live.connect({
        model: "gemini-3.1-flash-live-preview",
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Zephyr" } },
          },
          systemInstruction: systemInstruction,
          inputAudioTranscription: {enabled: false},
          outputAudioTranscription: {enabled: false},
        },
        callbacks: {
          onopen: () => {
            console.log("Live session opened");
            this.setupMicrophone();
          },
          onmessage: async (message: any) => {
            if (message.serverContent?.modelTurn?.parts) {
              for (const part of message.serverContent.modelTurn.parts) {
                if (part.inlineData?.data) {
                  const audioData = this.base64ToUint8Array(part.inlineData.data);
                  const pcmData = new Int16Array(
                    audioData.buffer,
                    audioData.byteOffset,
                    audioData.byteLength / 2
                  );

                  this.queueAudio(new Int16Array(pcmData));

                  // Accumulate energy for pre-graph long-term RMS (Phase 1)
                  if (!shouldPlayNoise) {
                    for (let i = 0; i < pcmData.length; i++) {
                      const sample = pcmData[i] / 32768.0;
                      this.preGraphSpeechSumSquares += sample * sample;
                      this.preGraphSpeechSampleCount++;
                    }
                  }
                }
              }
            }

            if (message.serverContent?.interrupted) {
              this.stopPlayback();
            }
          },
          onerror: (error: any) => {
            console.error("Live session error:", error);
            onError?.(error);
          },
          onclose: () => {
            console.log("Live session closed");
            this.stop();
            onClose?.();
          },
        },
      });

      this.session = await this.sessionPromise;
      
      // Trigger the model to start speaking first
      this.session.sendRealtimeInput({
        text: "Please start the conversation now based on your instructions."
      });

    } catch (error) {
      console.error("Failed to start live session:", error);
      throw error;
    }
  }

  private setupMicrophone() {
    if (!this.audioContext || !this.stream) return;

    this.source = this.audioContext.createMediaStreamSource(this.stream);
    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);

    this.processor.onaudioprocess = (e) => {
      if (!this.session) return; // Don't send if session is closed

      const inputData = e.inputBuffer.getChannelData(0);
      const pcmData = this.floatTo16BitPCM(inputData);
      const base64Data = this.uint8ArrayToBase64(new Uint8Array(pcmData.buffer));
      
      this.sessionPromise?.then(session => {
        if (this.session) {
          try {
            session.sendRealtimeInput({
              audio: { data: base64Data, mimeType: "audio/pcm;rate=24000" },
            });
          } catch (err) {
            console.error("Error sending audio to session:", err);
          }
        }
      });

      e.outputBuffer.getChannelData(0).fill(0); // samples mute always
    };

    this.source.connect(this.processor);
    this.processor.connect(this.audioContext.destination);

    // Record participant's voice into the transcript
    if (this.transcriptProcessor) {
      this.source.connect(this.transcriptProcessor);
    }

  }

  private queueAudio(pcmData: Int16Array) {
    this.audioQueue.push(pcmData);
    if (!this.isPlaying) {
      this.playNext();
    }
  }

  private async playNext() {
    if (this.audioQueue.length === 0 || !this.audioContext) {
      this.isPlaying = false;
      return;
    }

    this.isPlaying = true;
    const pcmData = this.audioQueue.shift()!;
    const floatData = new Float32Array(pcmData.length);
    for (let i = 0; i < pcmData.length; i++) {
      floatData[i] = pcmData[i] / 32768.0;
    }

    const buffer = this.audioContext.createBuffer(1, floatData.length, 24000);
    buffer.getChannelData(0).set(floatData);

    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;

    if (this.analyser) {
      source.connect(this.analyser);
    }

    // Always connect to a stable junction to avoid graph re-initialization per buffer
    if (this.voiceGainNode) {
      source.connect(this.voiceGainNode);
    }

    const startTime = Math.max(this.audioContext.currentTime, this.nextStartTime);
    try {
      source.start(startTime);
      this.nextStartTime = startTime + buffer.duration;
    } catch (e) {
      console.error("Error starting audio source:", e);
      this.isPlaying = false;
      this.playNext();
      return;
    }

    source.onended = () => {
      this.playNext();
    };
  }

  private stopPlayback() {
    this.audioQueue = [];
    this.nextStartTime = 0;
  }

  private floatTo16BitPCM(input: Float32Array): Int16Array {
    const output = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      const s = Math.max(-1, Math.min(1, input[i]));
      output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return output;
  }

  private base64ToUint8Array(base64: string): Uint8Array {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
  }

  private uint8ArrayToBase64(bytes: Uint8Array): string {
    let binary = "";
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  stop() {
    if (this.transcriptProcessor) {
      this.transcriptProcessor.disconnect();
      this.transcriptProcessor = null;
    }

    if (this.voiceRecordingProcessor) {
      this.voiceRecordingProcessor.disconnect();
      this.voiceRecordingProcessor = null;
    }

    if (this.noiseRecordingProcessor) {
      this.noiseRecordingProcessor.disconnect();
      this.noiseRecordingProcessor = null;
    }

    if (this.speechRMSProcessor) {
      this.speechRMSProcessor.disconnect();
      this.speechRMSProcessor = null;
    }

    if (this.noiseSource) {
      try {
        this.noiseSource.stop();
      } catch (e) {}
      this.noiseSource = null;
    }

    if (this.session) {
      try {
        this.session.close();
      } catch (e) {
        console.error("Error closing session:", e);
      }
      this.session = null;
    }

    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }

    if (this.source) {
      this.source.disconnect();
      this.source = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }

    if (this.convolver) {
      this.convolver.disconnect();
      this.convolver = null;
    }

    if (this.dryGainNode) {
      this.dryGainNode.disconnect();
      this.dryGainNode = null;
    }

    if (this.voiceGainNode) {
      this.voiceGainNode.disconnect();
      this.voiceGainNode = null;
    }

    if (this.audioContext) {
      if (this.audioContext.state !== "closed") {
        this.audioContext.close().catch((e) => console.error("Error closing AudioContext:", e));
      }
      this.audioContext = null;
    }

    this.audioQueue = [];
    this.isPlaying = false;
    this.nextStartTime = 0;
  }

  getByteFrequencyData(array: Uint8Array) {
    if (this.analyser) {
      this.analyser.getByteFrequencyData(array);
    }
  }

  getRecordings(): { transcript: Blob | null; voice: Blob | null; noise: Blob | null } {
    const sampleRate = this.audioContext?.sampleRate || 24000;

    const transcriptBlob = this.transcriptSamples.length > 0 
      ? new Blob([this.encodeWAV(this.mergeSamples(this.transcriptSamples), sampleRate)], { type: "audio/wav" })
      : null;

    const voiceBlob = this.voiceSamplesL.length > 0
      ? new Blob([this.encodeStereoWAV(this.mergeSamples(this.voiceSamplesL), this.mergeSamples(this.voiceSamplesR), sampleRate)], { type: "audio/wav" })
      : null;

    const noiseBlob = this.noiseSamplesL.length > 0
      ? new Blob([this.encodeStereoWAV(this.mergeSamples(this.noiseSamplesL), this.mergeSamples(this.noiseSamplesR), sampleRate)], { type: "audio/wav" })
      : null;

    return { transcript: transcriptBlob, voice: voiceBlob, noise: noiseBlob };
  }

  private mergeSamples(samples: Float32Array[]): Float32Array {
    const totalLength = samples.reduce((acc, val) => acc + val.length, 0);
    const result = new Float32Array(totalLength);
    let offset = 0;
    for (const sample of samples) {
      result.set(sample, offset);
      offset += sample.length;
    }
    return result;
  }

  private encodeWAV(samples: Float32Array, sampleRate: number) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (view: DataView, offset: number, string: string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true); // Block align
    view.setUint16(34, 16, true); // Bits per sample
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return buffer;
  }

  private encodeStereoWAV(leftSamples: Float32Array, rightSamples: Float32Array, sampleRate: number) {
    const length = Math.min(leftSamples.length, rightSamples.length);
    const buffer = new ArrayBuffer(44 + length * 4);
    const view = new DataView(buffer);

    const writeString = (view: DataView, offset: number, string: string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + length * 4, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 2, true); // Stereo
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 4, true);
    view.setUint16(32, 4, true); // Block align
    view.setUint16(34, 16, true); // Bits per sample
    writeString(view, 36, 'data');
    view.setUint32(40, length * 4, true);

    let offset = 44;
    for (let i = 0; i < length; i++) {
      let sL = Math.max(-1, Math.min(1, leftSamples[i]));
      view.setInt16(offset, sL < 0 ? sL * 0x8000 : sL * 0x7FFF, true);
      offset += 2;
      let sR = Math.max(-1, Math.min(1, rightSamples[i]));
      view.setInt16(offset, sR < 0 ? sR * 0x8000 : sR * 0x7FFF, true);
      offset += 2;
    }

    return buffer;
  }
}
