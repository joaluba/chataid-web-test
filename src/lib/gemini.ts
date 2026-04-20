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

export class LiveAudioSession {

  private static readonly SNR_DB = 15; 
  private static readonly NOISE_VOLUME = 1.0;
  
  // Noise Configuration
  private static readonly NOISE_TYPE: 'SPEECH_SHAPED' | 'FILE' = 'FILE';
  private static readonly NOISE_FILE_URL = "https://res.cloudinary.com/dqttqwfib/video/upload/v1776700340/cafe_noise_bin_pbsmfe.mp3";
  
  // Spatialization Configuration
  private static readonly CUSTOM_HRIR_URL = "https://res.cloudinary.com/dqttqwfib/video/upload/v1776699826/cafe_rir_bin_pnaczh.wav"; 
  // --------------------------

  private ai: any;
  private session: any;
  private audioContext: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private stream: MediaStream | null = null;
  private audioQueue: Int16Array[] = [];
  private isPlaying = false;
  private nextStartTime = 0;
  private noiseSource: AudioBufferSourceNode | null = null;
  private panner: PannerNode | null = null;
  private convolver: ConvolverNode | null = null;
  private voiceGainNode: GainNode | null = null;
  private analyser: AnalyserNode | null = null;
  private sessionPromise: Promise<any> | null = null;
  private recordedSamples: Float32Array[] = [];
  private recordingProcessor: ScriptProcessorNode | null = null;
  private recordingDestination: MediaStreamAudioDestinationNode | null = null;
  private hrtfType: 'BUILTIN' | 'CUSTOM' | 'NONE' = 'BUILTIN';

  constructor(apiKey: string) {
    if (!apiKey) {
      throw new Error("Gemini API key is required to start a session.");
    }
    this.ai = new GoogleGenAI({ apiKey });
  }

  private createSpeechShapedNoiseBuffer(context: AudioContext): AudioBuffer {
    const duration = 5.0;
    const sampleRate = context.sampleRate;
    const buffer = context.createBuffer(2, sampleRate * duration, sampleRate);
    
    for (let channel = 0; channel < 2; channel++) {
      const data = buffer.getChannelData(channel);
      // Generate white noise and apply a simple 1st order low-pass filter (integrator)
      // to approximate the -6dB/octave slope of LTASS above 500Hz
      let lastOut = 0;
      for (let i = 0; i < data.length; i++) {
        const white = Math.random() * 2 - 1;
        // Simple alpha filter: y[i] = alpha * x[i] + (1-alpha) * y[i-1]
        // For ~500Hz cutoff at 24kHz, alpha is approx 0.13
        const alpha = 0.13;
        const out = alpha * white + (1 - alpha) * lastOut;
        data[i] = out * 0.5; // Scale down to avoid clipping
        lastOut = out;
      }
    }
    return buffer;
  }

  private async loadNoiseFromFile(context: AudioContext, url: string): Promise<AudioBuffer> {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    return await context.decodeAudioData(arrayBuffer);
  }

  private async loadHRIR(context: AudioContext, url: string): Promise<AudioBuffer> {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    return await context.decodeAudioData(arrayBuffer);
  }

  private async startNoise() {
    if (!this.audioContext) return;
    
    let buffer: AudioBuffer;
    
    try {
      if (LiveAudioSession.NOISE_TYPE === 'FILE') {
        buffer = await this.loadNoiseFromFile(this.audioContext, LiveAudioSession.NOISE_FILE_URL);
      } else {
        buffer = this.createSpeechShapedNoiseBuffer(this.audioContext);
      }
    } catch (err) {
      console.error("Failed to load/create noise buffer, falling back to speech-shaped:", err);
      buffer = this.createSpeechShapedNoiseBuffer(this.audioContext);
    }

    this.noiseSource = this.audioContext.createBufferSource();
    this.noiseSource.buffer = buffer;
    this.noiseSource.loop = true;
    
    const gain = this.audioContext.createGain();
    gain.gain.value = LiveAudioSession.NOISE_VOLUME;
    
    this.noiseSource.connect(gain);
    gain.connect(this.audioContext.destination);
    
    //if (this.recordingDestination) {
    //  gain.connect(this.recordingDestination);
    //}
    
    this.noiseSource.start();
  }

  async start(params: {
    systemInstruction: string;
    shouldPlayNoise: boolean;
    hrtfType?: 'BUILTIN' | 'CUSTOM' | 'NONE';
    onError?: (error: any) => void;
    onClose?: () => void;
  }) {
    const { systemInstruction, shouldPlayNoise, hrtfType = 'BUILTIN', onError, onClose } = params;
    this.hrtfType = hrtfType;
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.audioContext = new AudioContext({ sampleRate: 24000 });
      
      if (this.audioContext.state === "suspended") {
        await this.audioContext.resume();
      }

      this.voiceGainNode = this.audioContext.createGain();
      // Phase 1 (no noise): gain = 1.0. Phase 2 (noise): gain based on SNR
      if (shouldPlayNoise) {
        const voiceGain = LiveAudioSession.NOISE_VOLUME * Math.pow(10, LiveAudioSession.SNR_DB / 20);
        this.voiceGainNode.gain.value = voiceGain;
      } else {
        this.voiceGainNode.gain.value = 1.0;
      }
      this.voiceGainNode.connect(this.audioContext.destination);

      // Setup Recording (WAV)
      this.recordingDestination = this.audioContext.createMediaStreamDestination();
      this.voiceGainNode.connect(this.recordingDestination);
      
      this.recordedSamples = [];
      this.recordingProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);
      const recordingSource = this.audioContext.createMediaStreamSource(this.recordingDestination.stream);
      recordingSource.connect(this.recordingProcessor);
      this.recordingProcessor.connect(this.audioContext.destination);
      
      this.recordingProcessor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        this.recordedSamples.push(new Float32Array(input));
      };

      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.connect(this.audioContext.destination);
      
      if (shouldPlayNoise) {
        this.startNoise();
      }

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
    };

    this.source.connect(this.processor);
    this.processor.connect(this.audioContext.destination);

    // to record participant's voice 
    if (this.recordingDestination) {
      this.source.connect(this.recordingDestination);
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

    // Use a stable spatializer node to avoid corruption/clicks
    if (this.voiceGainNode && !this.panner && !this.convolver && this.hrtfType !== 'NONE') {
      if (this.hrtfType === 'BUILTIN') {
        this.panner = this.audioContext.createPanner();
        this.panner.panningModel = 'HRTF';
        this.panner.distanceModel = 'inverse';
        this.panner.positionX.value = 0;
        this.panner.positionY.value = 0;
        this.panner.positionZ.value = -1;
        this.panner.connect(this.voiceGainNode);
      } else if (this.hrtfType === 'CUSTOM') {
        this.convolver = this.audioContext.createConvolver();
        try {
          const hrirBuffer = await this.loadHRIR(this.audioContext, LiveAudioSession.CUSTOM_HRIR_URL);
          this.convolver.buffer = hrirBuffer;
          this.convolver.connect(this.voiceGainNode);
        } catch (err) {
          console.error("Failed to load custom HRIR, falling back to BUILTIN panner:", err);
          this.panner = this.audioContext.createPanner();
          this.panner.panningModel = 'HRTF';
          this.panner.connect(this.voiceGainNode);
        }
      }
    }

    if (this.panner) {
      source.connect(this.panner);
    } else if (this.convolver) {
      source.connect(this.convolver);
    } else {
      // No spatialization
      if (this.voiceGainNode) {
        source.connect(this.voiceGainNode);
      }
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
    // In a real implementation, we might want to stop the current source node
    // but ScriptProcessor doesn't give us easy access to the active source nodes.
    // For simplicity, we just clear the queue.
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
    if (this.recordingProcessor) {
      this.recordingProcessor.disconnect();
      this.recordingProcessor = null;
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

    if (this.panner) {
      this.panner.disconnect();
      this.panner = null;
    }

    if (this.convolver) {
      this.convolver.disconnect();
      this.convolver = null;
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

  getRecording(): Blob | null {
    if (this.recordedSamples.length === 0) return null;
    
    const totalLength = this.recordedSamples.reduce((acc, val) => acc + val.length, 0);
    const result = new Float32Array(totalLength);
    let offset = 0;
    for (const sample of this.recordedSamples) {
      result.set(sample, offset);
      offset += sample.length;
    }

    const wavBuffer = this.encodeWAV(result, this.audioContext?.sampleRate || 24000);
    return new Blob([wavBuffer], { type: "audio/wav" });
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
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return buffer;
  }
}
