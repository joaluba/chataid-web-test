import React, { useState, useEffect, useRef } from "react";
import { Mic, MicOff, Coffee, Music, Clock, Info, Loader2 } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { LiveAudioSession, INSTRUCTION_PROMPT, EXPERIMENT_PROMPT } from "../lib/gemini";
import JSZip from "jszip";
import { doc, setDoc } from "firebase/firestore";
import { db, handleFirestoreError, OperationType } from "../lib/firebase";

let APP_VER_INFO = `App version: V0.1.4(ZIP+FIREBASE checkpoint) SNR=${LiveAudioSession.SNR_DB}`;

enum ExperimentStep {
  AUTH = 0,
  INTRO = 1,
  PARTICIPANT_FORM = 2,
  TRAINING_EXPLANATION = 3,
  TRAINING_PHASE = 4,
  INITIAL_QUESTIONNAIRE_EXPLANATION = 5,
  INITIAL_QUESTIONNAIRE = 6,
  EXPERIMENT_EXPLANATION = 7,
  EXPERIMENT_PHASE = 8,
  FINAL_QUESTIONNAIRE_EXPLANATION = 9,
  FINAL_QUESTIONNAIRE = 10,
  GOODBYE = 11,
}

const QUESTIONNAIRE_STRUCTURE = [
  {
    id: "cat1",
    title: "TTS voice quality",
    description: "Rate the quality of the generated agent's voice.",
    questions: [
      { id: "effort", text: "Please rate the degree of effort you had to make to understand the message.", min: "1 = Impossible even with much effort", max: "7 = No effort required" },
      { id: "singleWords", text: "Were single words hard to understand?", min: "1 = All words hard to understand", max: "7 = All words easy to understand" },
      { id: "clearlyDistinguishable", text: "Were the speech sounds clearly distinguishable?", min: "1 = Not at all clear", max: "7 = Very clear" },
      { id: "preciseArticulation", text: "Was the articulation of speech sounds precise?", min: "1 = Slurried or imprecise", max: "7 = Precise" },
      { id: "pleasantVoice", text: "Was the voice you heard pleasant to listen to?", min: "1 = Very unpleasant", max: "7 = Very pleasant" },
      { id: "naturalVoice", text: "Did the voice sound natural?", min: "1 = Very unnatural", max: "7 = Very natural" },
      { id: "humanLike", text: "To what extent did the voice sound like a human?", min: "1 = Nothing like a human", max: "7 = Just like a human" },
      { id: "harshVoice", text: "Did the voice sound harsh, raspy, or strained?", min: "1 = Significantly harsh/raspy", max: "7 = Normal quality" },
      { id: "emphasis", text: "Did emphasis of important words occur?", min: "1 = Incorrect emphasis", max: "7 = Excellent use of emphasis" },
      { id: "naturalRhythm", text: "Did the rhythm of the speech sound natural?", min: "1 = Unnatural or mechanical", max: "7 = Natural rhythm" },
      { id: "smoothIntonation", text: "Did the intonation pattern of sentences sound smooth and natural?", min: "1 = Abrupt or abnormal", max: "7 = Smooth or normal" },
      { id: "trustworthy", text: "Did the voice appear to be trustworthy?", min: "1 = Not at all trustworthy", max: "7 = Very trustworthy" },
      { id: "confident", text: "Did the voice suggest a confident speaker?", min: "1 = Not at all confident", max: "7 = Very confident" },
      { id: "enthusiastic", text: "Did the voice seem to be enthusiastic?", min: "1 = Not at all enthusiastic", max: "7 = Very enthusiastic" },
      { id: "persuasive", text: "Was the voice persuasive?", min: "1 = Not at all persuasive", max: "7 = Very persuasive" },
    ]
  },
  {
    id: "cat2",
    title: "Agent interaction quality",
    description: "Rate the quality of the interaction with the voice agent, focusing on timing, turn-taking, delays, pauses, interruptions, and conversational flow.",
    questions: [
      { id: "delayAcceptable", text: "The delay between my speech and the agent's reply felt acceptable.", min: "1 = Strongly disagree", max: "7 = Strongly agree" },
      { id: "freeFlowing", text: "How free-flowing did the conversation feel?", min: "1 = Not at all free-flowing", max: "7 = Completely free-flowing" },
      { id: "naturalConv", text: "How natural did the conversation feel?", min: "1 = Not at all natural", max: "7 = Completely natural" },
      { id: "awkwardPauses", text: "How often during the conversation did you experience awkward pauses, interruptions, or talking over the agent?", min: "1 = Very frequently", max: "7 = Never" },
    ]
  },
  {
    id: "cat3",
    title: "Passive listening difficulty",
    description: "Rate how difficult it was to UNDERSTAND the agent's voice in a noisy café-like acoustic scene.",
    questions: [
      { id: "understandAmount", text: "How much of the agent's speech could you understand in this conversation?", min: "1 = Not at all", max: "7 = Everything" },
      { id: "understandEffort", text: "How much effort did it take you to understand the agent's speech?", min: "1 = No effort at all", max: "7 = Extreme effort" },
    ]
  },
  {
    id: "cat4",
    title: "Communication difficulty",
    description: "Rate how difficult it was to communicate (INTERACT) with the agent in the noisy café-like acoustic scene.",
    questions: [
      { id: "stressful", text: "How stressful was it to have this conversation?", min: "1 = Not stressful at all", max: "7 = Extremely stressful" },
      { id: "noiseDifficulty", text: "The noise made it hard to have a smooth back-and-forth exchange with the agent.", min: "1 = Strongly disagree", max: "7 = Strongly agree" },
      { id: "commSuccess", text: "How successful was the communication overall?", min: "1 = Completely unsuccessful", max: "7 = Completely successful" },
      { id: "askRepeat", text: "How often did you have to ask the agent to repeat or rephrase something?", min: "1 = Never", max: "7 = Very frequently" },
    ]
  },
  {
    id: "cat5",
    title: "Task ecological validity",
    description: "Rate how realistic and meaningful the role-play task felt as a situation you might encounter in everyday life. Focus on the task and your engagement with it, not on the voice quality or background noise.",
    questions: [
      { id: "relevance", text: "How relevant was this listening situation to your everyday life?", min: "1 = Not at all relevant", max: "7 = Extremely relevant" },
      { id: "imagineDoing", text: "The task I was asked to complete is something I could imagine doing in a real café.", min: "1 = Strongly disagree", max: "7 = Strongly agree" },
      { id: "engaged", text: "How engaged did you feel in the role-play task?", min: "1 = Not at all engaged", max: "7 = Extremely engaged" },
    ]
  },
  {
    id: "cat6",
    title: "Acoustic scene ecological validity",
    description: "Rate how realistic and immersive the café-like acoustic environment sounded. Focus on the background noise, the sense of being in a real café, and whether the target voice fit naturally into that environment.",
    questions: [
      { id: "realCafe", text: "The acoustic environment sounded like a real café I might find myself in.", min: "1 = Strongly disagree", max: "7 = Strongly agree" },
      { id: "immersive", text: "How immersive did the acoustic environment feel?", min: "1 = Not at all immersive", max: "7 = Extremely immersive" },
      { id: "realisticNoise", text: "How realistic was the background noise for a café environment?", min: "1 = Not at all realistic", max: "7 = Completely realistic" },
      { id: "realisticVoice", text: "How realistic was the target voice sound for a café environment?", min: "1 = Not at all realistic", max: "7 = Completely realistic" },
    ]
  }
];

const AudioVisualizer = ({ session }: { session: LiveAudioSession | null }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(null);

  useEffect(() => {
    if (!session || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const bufferLength = 128;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animationRef.current = requestAnimationFrame(draw);
      session.getByteFrequencyData(dataArray);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 2.5;
      let barHeight;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        // Standard normalization: dataArray[i] is 0-255. 
        // We map 255 to the full canvas height.
        // This ensures bars are proportional to volume and don't all hit the ceiling.
        barHeight = (dataArray[i] / 255) * canvas.height;

        ctx.fillStyle = "black";
        ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);

        x += barWidth + 1;
      }
    };

    draw();

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [session]);

  return (
    <canvas 
      ref={canvasRef} 
      width={300} 
      height={40} 
      className="w-full h-10 bg-white"
    />
  );
};

const INITIAL_TASKS = [
{ id: 1, text: "Price of a coffee with milk", understanding: "" },
{ id: 2, text: "Milk options available", understanding: "" },
{ id: 3, text: "Is vegan milk more expensive?", understanding: "" },
{ id: 4, text: "What is the cafe's specialty cake?", understanding: "" },
{ id: 5, text: "Wi-Fi network name", understanding: "" },
{ id: 6, text: "Wi-Fi password", understanding: "" },
{ id: 7, text: "Maximum table usage duration", understanding: "" },
{ id: 8, text: "Evening event", understanding: "" },
{ id: 9, text: "Artist's name", understanding: "" },
{ id: 10, text: "Cafe closing time", understanding: "" },
];

const TRAINING_TASKS_INITIAL = [
  { id: 101, text: "Price of a single metro ticket", understanding: "" },
  { id: 102, text: "Current sea water temperature", understanding: "" },
  { id: 103, text: "Which museums have free admission today", understanding: "" },
  { id: 104, text: "Tourist office closing time", understanding: "" },
];

const ProgressBar = ({ step }: { step: ExperimentStep }) => {
  if (step === ExperimentStep.AUTH || step === ExperimentStep.GOODBYE) return null;
  const total = 10;
  const progress = (step / total) * 100;

  return (
    <div className="fixed top-0 left-0 w-full h-1 bg-gray-100 z-50">
      <div 
        className="h-full bg-black transition-all duration-500 ease-out" 
        style={{ width: `${progress}%` }}
      />
    </div>
  );
};

export default function VoiceAgent() {
  const [currentStep, setCurrentStep] = useState<ExperimentStep>(ExperimentStep.AUTH);
  const [isActive, setIsActive] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isCooldown, setIsCooldown] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionRef = useRef<LiveAudioSession | null>(null);



  // User data form state
  const [isUploading, setIsUploading] = useState(false);
  
  const [initialQuestionnaireAnswers, setInitialQuestionnaireAnswers] = useState<Record<string, number>>(() => {
    const initial = {};
    QUESTIONNAIRE_STRUCTURE.filter(c => c.id === "cat1" || c.id === "cat2").flatMap(c => c.questions).forEach(q => {
      // @ts-ignore
      initial[q.id] = 4;
    });
    return initial;
  });
  const [finalQuestionnaireAnswers, setFinalQuestionnaireAnswers] = useState<Record<string, number>>(() => {
    const initial = {};
    QUESTIONNAIRE_STRUCTURE.filter(c => c.id !== "cat1" && c.id !== "cat2").flatMap(c => c.questions).forEach(q => {
      // @ts-ignore
      initial[q.id] = 4;
    });
    return initial;
  });
  
  const [participantAlias, setParticipantAlias] = useState("Test_User");
  const [participantAge, setParticipantAge] = useState("30");
  const [gender, setGender] = useState("Female");
  const [isNativeSpeaker, setIsNativeSpeaker] = useState("no");
  const [hearingStatus, setHearingStatus] = useState("Normal hearing");
  const [isListeningExpert, setIsListeningExpert] = useState("no");
  const [usingHeadphones, setUsingHeadphones] = useState("yes");
  const [hasConsented, setHasConsented] = useState(false);
  const [passphrase, setPassphrase] = useState("");
  const [conditionPhrase, setConditionPhrase] = useState("");
  const [passphraseError, setPassphraseError] = useState(false);

  const [introRead, setIntroRead] = useState(false);
  const [tasks, setTasks] = useState(INITIAL_TASKS);
  const [trainingTasks, setTrainingTasks] = useState(TRAINING_TASKS_INITIAL);

  const updateTask = (id: number, field: "understanding", value: string) => {
    setTasks(prev => prev.map(t => t.id === id ? { ...t, [field]: value } : t));
  };

  const updateTrainingTask = (id: number, field: "understanding", value: string) => {
    setTrainingTasks(prev => prev.map(t => t.id === id ? { ...t, [field]: value } : t));
  };

  const toggleSession = async () => {
    if (isCooldown) return;

    if (isActive) {
      sessionRef.current?.stop();
      setIsActive(false);
      
      setIsCooldown(true);
      setTimeout(() => setIsCooldown(false), 1000);
    } else {
      setIsConnecting(true);
      setError(null);
      try {
        if (!sessionRef.current) {
          const apiKey = process.env.GEMINI_API_KEY;
          if (!apiKey) {
            throw new Error("No Gemini API key provided.");
          }
          sessionRef.current = new LiveAudioSession(apiKey);
        }
        
        const isTraining = currentStep === ExperimentStep.TRAINING_PHASE;
        await sessionRef.current.start({
          systemInstruction: isTraining ? INSTRUCTION_PROMPT : EXPERIMENT_PROMPT,
          shouldPlayNoise: !isTraining,
          onError: (err) => {
            const errorMessage = err?.message || String(err);
            if (errorMessage.toLowerCase().includes("quota")) {
              setError("Gemini API quota reached. Please wait 60 seconds.");
              setIsCooldown(true);
              setTimeout(() => setIsCooldown(false), 10000);
            } else {
              setError("Something went wrong with the connection: " + errorMessage);
            }
            setIsActive(false);
          },
          onClose: () => {
            setIsActive(false);
          },
        });
        setIsActive(true);
      } catch (err: any) {
        setIsConnecting(false);
        setError("Could not access microphone or connect to Gemini: " + (err?.message || String(err)));
      } finally {
        setIsConnecting(false);
      }
    }
  };

  const nextStep = () => {
    if (isActive) {
      sessionRef.current?.stop();
      setIsActive(false);
    }
    setCurrentStep(prev => prev + 1);
  };

  const finalizeSession = async () => {
    setIsUploading(true);
    setError(null);

    try {
      const docId = `${participantAlias}_${new Date().getTime()}`;

      // 1. Prepare Full Data Dictionary
      const fullDataDictionary = {
        exportId: docId,
        participant: {
          alias: participantAlias,
          age: participantAge,
          gender,
          isNativeSpeaker,
          hearingStatus,
          isListeningExpert,
          usingHeadphones
        },
        metadata: {
          appVersion: APP_VER_INFO,
          timestamp: new Date().toISOString()
        },
        training_userinput: trainingTasks.reduce((acc: any, t) => ({ ...acc, [t.text]: t.understanding }), {}),
        training_questionnaire: initialQuestionnaireAnswers,
        experiment_userinput: tasks.reduce((acc: any, t) => ({ ...acc, [t.text]: t.understanding }), {}),
        experiment_questionnaire: finalQuestionnaireAnswers
      };

      // 1b. Save JSON data securely to Firebase Firestore
      try {
        await setDoc(doc(db, "sessions", docId), fullDataDictionary);
      } catch (firestoreError) {
        console.error("Failed to save session to Firestore:", firestoreError);
        handleFirestoreError(firestoreError, OperationType.WRITE, `sessions/${docId}`);
      }

      // 2. Create ZIP and trigger download
      const zip = new (JSZip as any)();
      
      // Add JSON data
      zip.file(`experiment_data.json`, JSON.stringify(fullDataDictionary, null, 2));

      // Add audio recordings if available
      try {
        const recordings = sessionRef.current?.getRecordings();
        if (recordings) {
          if (recordings.training) zip.file(`transcript_training.wav`, recordings.training);
          if (recordings.main) zip.file(`transcript_experiment.wav`, recordings.main);
          if (recordings.voice) zip.file(`agent_experiment.wav`, recordings.voice);
          if (recordings.noise) zip.file(`noise_experiment.wav`, recordings.noise);
        }
      } catch (recordingErr) {
        console.error("Failed to get recordings:", recordingErr);
      }

      // Generate the single ZIP file
      const content = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(content);
      const a = document.createElement("a");
      a.href = url;
      a.download = `experiment_${docId}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      
      nextStep();
    } catch (err: any) {
      console.error("Critical finalize error:", err);
      setError("An error occurred during export: " + (err.message || String(err)));
    } finally {
      setIsUploading(false);
    }
  };

  useEffect(() => {
    return () => {
      sessionRef.current?.stop();
    };
  }, []);





  // --- 0. AUTH ---
  if (currentStep === ExperimentStep.AUTH) {
    const handleAuthorize = () => {
      if (passphrase === "ChatAid2026") {
        if (conditionPhrase) {
          const lowerPhrase = conditionPhrase.toLowerCase();
          const snrIndex = lowerPhrase.indexOf("snr=");
          if (snrIndex !== -1) {
            const snrPart = conditionPhrase.slice(snrIndex + 4).trim();
            const numVal = parseFloat(snrPart);
            if (!isNaN(numVal)) {
              LiveAudioSession.SNR_DB = numVal;
              APP_VER_INFO = `App version: V0.1.4(ZIP+FIREBASE checkpoint) SNR=${LiveAudioSession.SNR_DB}`;
            }
          }
        }
        nextStep();
      } else {
        setPassphraseError(true);
      }
    };

    return (
      <div className="min-h-screen bg-white flex flex-col items-center justify-center p-6 font-sans text-black relative">
        <div className="absolute bottom-4 left-4 text-xs text-gray-500 font-sans uppercase tracking-wider">
          {APP_VER_INFO}
        </div>
        <div className="w-full max-w-sm p-10 text-center space-y-8">
          <h1 className="text-[25px] font-medium text-black">Restricted access</h1>
          <p className="text-black text-base italic leading-relaxed">
            Please enter the passphrase and condition phrase provided by the researcher to continue.
          </p>
          <div className="space-y-4">
            <input
              type="password"
              value={passphrase}
              onChange={(e) => {
                setPassphrase(e.target.value);
                setPassphraseError(false);
              }}
              onKeyDown={(e) => e.key === "Enter" && handleAuthorize()}
              className={`w-full text-center text-base bg-transparent border-b ${passphraseError ? "border-red-500" : "border-black"} pb-1 focus:outline-none text-black font-sans`}
              placeholder="Enter passphrase"
              autoFocus
            />
            <input
              type="text"
              value={conditionPhrase}
              onChange={(e) => {
                setConditionPhrase(e.target.value);
              }}
              onKeyDown={(e) => e.key === "Enter" && handleAuthorize()}
              className="w-full text-center text-base bg-transparent border-b border-black pb-1 focus:outline-none text-black font-sans"
              placeholder="Enter condition phrase"
            />
            {passphraseError && (
              <p className="text-base text-red-500 italic">Incorrect passphrase. Please try again.</p>
            )}
          </div>
          <button
            onClick={handleAuthorize}
            className="w-full py-3 bg-white text-black border border-black rounded-lg text-base font-sans hover:bg-gray-50 transition-colors"
          >
            Enter
          </button>
        </div>
      </div>
    );
  }

  // --- 1. INTRO SCREEN ---
  if (currentStep === ExperimentStep.INTRO) {
    return (
      <div className="min-h-screen bg-white flex flex-col items-center justify-start p-6 pb-24 font-sans text-black relative pt-32">
        <ProgressBar step={currentStep} />
        <div className="absolute bottom-4 left-4 text-xs text-gray-500 font-sans uppercase tracking-wider">
          {APP_VER_INFO}
        </div>
        <h1 className="text-[25px] font-medium tracking-tight mb-12 text-center max-w-2xl">Welcome to our study!</h1>
        
        <div className="w-full max-w-2xl p-10 pt-0 space-y-10">
          <div className="space-y-8 text-base text-black font-sans leading-relaxed">
            <div className="space-y-2">
              <h2 className="text-xs text-gray-500 font-sans uppercase tracking-wider font-bold">Context</h2>
              <p>
                We are a team of audio researchers at Eurecat Technology Center in Barcelona and we are investigating whether AI-based conversational agents can serve as reliable conversation partners in speech communication tests. Traditionally, such tests rely on human interlocutors, which limits how reproducible and scalable the evaluations can be. By validating AI agents in this role, we hope to contribute to more flexible and consistent methods for assessing speech communication in challenging listening conditions.
              </p>
            </div>

            <div className="space-y-4">
              <h2 className="text-xs text-gray-500 font-sans uppercase tracking-wider font-bold">Your role</h2>
              <p>
                Today you will participate in a pilot test, which will help us obtain a preliminary validation of our ideas. For that, we will ask you to engage in a role-play interaction with an AI voice agent. During the conversation, your task will be to gather specific pieces of information from the agent. The conversation will take place under carefully selected acoustic conditions, designed to reflect realistic everyday listening situations. After the interaction, you will be asked to answer a questionnaire about your experience.
              </p>
            </div>

            <div className="space-y-4">
              <h2 className="text-xs text-gray-500 font-sans uppercase tracking-wider font-bold">What will happen step by step</h2>
              <p>This experiment will consist of 4 parts:</p>
              <div className="space-y-4 ml-4">
                <p><strong>1/4 -&gt; Training phase</strong>, in which you will practice interacting with the agent in silence.</p>
                <p><strong>2/4 -&gt; Initial questionnaire</strong>, in which you will answer a set of questions about your first impressions about the agent.</p>
                <p><strong>3/4 -&gt; Main experiment phase</strong>, in which you will be interacting with the agent in more realistic acoustic conditions.</p>
                <p><strong>4/4 -&gt; Final questionnaire</strong>, in which you will answer another set of questions about the communication scenario.</p>
              </div>
            </div>

            <div className="space-y-4">
              <h2 className="text-xs text-gray-500 font-sans uppercase tracking-wider font-bold">What will be recorded</h2>
              <p>The following data will be recorded:</p>
              <ul className="list-disc ml-6 space-y-1">
                <li>Your conversations with the agent</li>
                <li>Your written responses in the information collection task</li>
                <li>Your ratings in the questionnaires</li>
              </ul>
              <p className="mt-4">
                You will download this data to your computer at the end of the session and we will ask you to send it to us by e-mail.
              </p>
            </div>

            <div className="space-y-2 pt-4 border-t border-dotted border-black">
              <h2 className="text-xs text-gray-500 font-sans uppercase tracking-wider font-bold">Before you begin</h2>
              <ul className="list-disc ml-6 space-y-1">
                <li>Find a calm environment where you can listen and speak freely without interruptions.</li>
                <li>Please make sure you are using <strong>headphones</strong> – important.</li>
                <li>Please read the consent statement below carefully. You can only proceed once you have given your informed consent.</li>
              </ul>
            </div>
          </div>

          <div className="pt-4 flex flex-col gap-4">
            <h2 className="text-xs text-gray-500 font-sans uppercase tracking-wider font-bold">Consent to participate</h2>
            <div className="flex items-start gap-4">
              <input
                type="checkbox"
                id="consent"
                checked={hasConsented}
                onChange={(e) => setHasConsented(e.target.checked)}
                className="mt-1 accent-black cursor-pointer w-4 h-4 shrink-0"
              />
              <label htmlFor="consent" className="text-[10px] text-gray-500 font-sans leading-normal cursor-pointer block">
                I consent to take part in this study and understand that my spoken responses and interaction with the voice agent will be recorded to produce a transcript and analyse the dialogue, including turn-taking and response patterns. The recording will not be used to identify me, analyse my voice characteristics, or perform biometric identification. To develop and run this app, the researchers use service providers such as the Gemini Speech API for the voice agent, Vercel for website deployment, and Firebase for text data storage; these providers may have access to some information extracted from this interaction. The data will be processed securely and pseudonymised where possible. My participation is voluntary, and I may stop at any time or request withdrawal/deletion of my identifiable data by contacting Joanna Luberadzka [joanna.luberadzka@eurecat.org], unless the data have already been anonymised or included in aggregated analysis. By continuing, I confirm that I have read the study information and give consent to this recording and processing for research purposes.
              </label>
            </div>
          </div>

          <p className="mt-8 text-[25px] font-medium tracking-tight text-center">Thank you and enjoy the experience!</p>

          <div className="pt-6 flex justify-center">
            <button 
              onClick={nextStep} 
              disabled={!hasConsented}
              className="px-12 py-3 bg-white text-black border border-black rounded-lg text-base font-sans hover:bg-gray-50 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    );
  }

  // --- 2. PARTICIPANT FORM ---
  if (currentStep === ExperimentStep.PARTICIPANT_FORM) {
    const isFormComplete = participantAlias && participantAge && gender && isNativeSpeaker && hearingStatus && isListeningExpert && usingHeadphones;

    return (
      <div className="min-h-screen bg-white flex flex-col items-center justify-start p-6 pb-24 font-sans text-black relative pt-32">
        <ProgressBar step={currentStep} />
        <div className="absolute bottom-4 left-4 text-xs text-gray-500 font-sans uppercase tracking-wider">
          {APP_VER_INFO}
        </div>
        <h1 className="text-[25px] font-medium mb-12 text-center tracking-tight">Participant information</h1>
        <div className="w-full max-w-xl p-8 md:p-10 pt-0">
          <div className="space-y-6">
            <div className="flex flex-col gap-2">
              <label className="text-xs text-gray-500 font-sans uppercase tracking-wider font-bold">Participant alias</label>
              <input
                type="text"
                value={participantAlias}
                onChange={(e) => setParticipantAlias(e.target.value)}
                placeholder="Enter your alias"
                className="w-full text-base text-black font-sans bg-transparent border-b border-dotted border-black pb-1 focus:outline-none"
              />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs text-gray-500 font-sans uppercase tracking-wider font-bold">Age</label>
              <input
                type="number"
                value={participantAge}
                onChange={(e) => setParticipantAge(e.target.value)}
                placeholder="Enter your age"
                className="w-full text-base text-black font-sans bg-transparent border-b border-dotted border-black pb-1 focus:outline-none"
              />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs text-gray-500 font-sans uppercase tracking-wider font-bold">Gender</label>
              <select
                value={gender}
                onChange={(e) => setGender(e.target.value)}
                className="w-full text-base text-black font-sans bg-transparent border-b border-dotted border-black pb-1 focus:outline-none"
              >
                <option value="">Select gender</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Non-binary">Non-binary</option>
                <option value="Prefer to self-describe">Prefer to self-describe</option>
              </select>
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs text-gray-500 font-sans uppercase tracking-wider font-bold">Native English speaker</label>
              <div className="flex gap-4">
                <label className="flex items-center gap-2 text-base text-black font-sans cursor-pointer">
                  <input type="radio" name="nativeSpeaker" value="yes" checked={isNativeSpeaker === "yes"} onChange={(e) => setIsNativeSpeaker(e.target.value)} className="accent-black" /> Yes
                </label>
                <label className="flex items-center gap-2 text-base text-black font-sans cursor-pointer">
                  <input type="radio" name="nativeSpeaker" value="no" checked={isNativeSpeaker === "no"} onChange={(e) => setIsNativeSpeaker(e.target.value)} className="accent-black" /> No
                </label>
              </div>
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs text-gray-500 font-sans uppercase tracking-wider font-bold">Hearing status</label>
              <select
                value={hearingStatus}
                onChange={(e) => setHearingStatus(e.target.value)}
                className="w-full text-base text-black font-sans bg-transparent border-b border-dotted border-black pb-1 focus:outline-none"
              >
                <option value="">Select status</option>
                <option value="Normal hearing">Normal hearing</option>
                <option value="Hearing impaired">Hearing impaired</option>
                <option value="Not sure">Not sure</option>
              </select>
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs text-gray-500 font-sans uppercase tracking-wider font-bold">Listening expert</label>
              <div className="flex gap-4">
                <label className="flex items-center gap-2 text-base text-black font-sans cursor-pointer">
                  <input type="radio" name="listeningExpert" value="yes" checked={isListeningExpert === "yes"} onChange={(e) => setIsListeningExpert(e.target.value)} className="accent-black" /> Yes
                </label>
                <label className="flex items-center gap-2 text-base text-black font-sans cursor-pointer">
                  <input type="radio" name="listeningExpert" value="no" checked={isListeningExpert === "no"} onChange={(e) => setIsListeningExpert(e.target.value)} className="accent-black" /> No
                </label>
              </div>
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs text-gray-500 font-sans uppercase tracking-wider font-bold">Are you using headphones?</label>
              <div className="flex gap-4">
                <label className="flex items-center gap-2 text-base text-black font-sans cursor-pointer">
                  <input type="radio" name="usingHeadphones" value="yes" checked={usingHeadphones === "yes"} onChange={(e) => setUsingHeadphones(e.target.value)} className="accent-black" /> Yes
                </label>
                <label className="flex items-center gap-2 text-base text-black font-sans cursor-pointer">
                  <input type="radio" name="usingHeadphones" value="no" checked={usingHeadphones === "no"} onChange={(e) => setUsingHeadphones(e.target.value)} className="accent-black" /> No
                </label>
              </div>
            </div>
          </div>

          <div className="mt-10 flex justify-center">
            <button
              onClick={nextStep}
              disabled={!isFormComplete}
              className="px-12 py-3 bg-white text-black border border-black rounded-lg text-base font-sans hover:bg-gray-50 transition-colors disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed"
            >
              Continue
            </button>
          </div>
        </div>
      </div>
    );
  }

  // --- 3. TRAINING EXPLANATION ---
  if (currentStep === ExperimentStep.TRAINING_EXPLANATION) {
    return (
      <div className="min-h-screen bg-white flex flex-col items-center justify-start p-6 pb-24 font-sans text-black relative pt-32">
        <ProgressBar step={currentStep} />
        <div className="absolute bottom-4 left-4 text-xs text-gray-500 font-sans uppercase tracking-wider">
          {APP_VER_INFO}
        </div>
        <h1 className="text-[25px] font-medium tracking-tight mb-12">1/4 Training</h1>
        <div className="text-center space-y-8">
          <div className="space-y-4">
            <p className="text-base text-black font-sans">In this phase you should:</p>
            <ul className="text-left max-w-md mx-auto space-y-1 inline-block list-disc pl-5 text-base text-black font-sans">
              <li>Test your microphone and headphones</li>
              <li>Adjust the loudness to the comfortable level</li>
              <li>Practice communication role-play with the agent (Tourist Office) </li>
              <li>Practice answering in the user input field</li>
              <li>Pay attention to the agent’s voice <br/></li>
            </ul>
          </div>
          <button onClick={nextStep} className="px-12 py-3 bg-white text-black border border-black rounded-lg text-base font-sans hover:bg-gray-50 transition-all">
            Begin training
          </button>
        </div>
      </div>
    );
  }

  // --- 4. TRAINING PHASE ---
  if (currentStep === ExperimentStep.TRAINING_PHASE) {
    return (
      <div className="min-h-screen bg-white flex flex-col items-center justify-start p-6 pb-24 font-sans text-black gap-12 relative pt-32">
        <ProgressBar step={currentStep} />
        <div className="absolute bottom-4 left-4 text-xs text-gray-500 font-sans uppercase tracking-wider">
          {APP_VER_INFO}
        </div>
        <h1 className="text-[25px] font-medium tracking-tight">1/4 Training</h1>
        
        <div className="flex flex-col items-center gap-12 w-full max-w-xl">
          <button
            onClick={toggleSession}
            disabled={isConnecting || isCooldown}
            className="px-8 py-3 bg-white border border-black rounded-lg text-base font-sans hover:bg-gray-50 transition-colors min-w-[124px] text-black"
          >
            {isConnecting ? <Loader2 className="animate-spin" /> : isActive ? "Stop session" : "Start session"}
          </button>
          
          <div className="relative w-full aspect-[16/4] border border-black rounded-[24px] overflow-hidden bg-white group select-none">
            <img
              src="https://images.unsplash.com/photo-1766098556973-2208e32fad19?q=80&w=1064&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
              alt="Acoustic experiment scene"
              className={`absolute inset-0 w-full h-full object-cover object-[center_75%] transition-all duration-300 ${
                isActive 
                  ? "opacity-60 grayscale-0 brightness-100" 
                  : "opacity-35 grayscale brightness-105"
              }`}
              referrerPolicy="no-referrer"
            />
            {isActive ? (
              <div className="absolute inset-0 bg-white/10 flex flex-col items-center justify-center p-4 z-10">
                <div className="w-48 p-1 bg-white border border-black">
                  <AudioVisualizer session={sessionRef.current} />
                </div>
              </div>
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-xs text-gray-400 font-mono uppercase tracking-widest bg-gray-50/20">
                Audio scene off
              </div>
            )}
          </div>

          <div className="w-full p-10 space-y-8 bg-[#FFFDC0] rounded-[40px] border border-black shadow-sm">
            <div className="space-y-2 border-b border-black pb-4">
              <div className="grid grid-cols-2 gap-8 font-sans text-xs text-gray-500 uppercase tracking-wider">
                <div className="font-bold">Information to collect:</div>
                <div className="font-bold">What I understood:</div>
              </div>
            </div>
            <div className="space-y-6 text-black">
              {trainingTasks.map(task => (
                <div key={task.id} className="grid grid-cols-2 gap-8 border-b border-dotted border-gray-300 pb-2 items-end">
                  <div className="text-base font-sans">{task.text}</div>
                  <input
                    type="text"
                    value={task.understanding}
                    onChange={(e) => updateTrainingTask(task.id, "understanding", e.target.value)}
                    className="bg-transparent text-base font-sans text-black focus:outline-none placeholder:text-gray-300 italic"
                    placeholder="Type here..."
                  />
                </div>
              ))}
            </div>
          </div>

          <div className="flex flex-col items-center gap-4">
            <button 
              onClick={() => {
                if (isActive) {
                  if (window.confirm("Are you sure you want to close the session and continue?")) {
                    nextStep();
                  }
                } else {
                  nextStep();
                }
              }} 
              className="px-12 py-3 bg-white text-black border border-black rounded-lg text-base font-sans hover:bg-gray-50"
            >
              Continue
            </button>
            <p className="text-xs text-black font-sans">Ensure the audio session is stopped before continuing</p>
          </div>
        </div>
      </div>
    );
  }

  // --- 5. INITIAL QUESTIONNAIRE EXPLANATION ---
  if (currentStep === ExperimentStep.INITIAL_QUESTIONNAIRE_EXPLANATION) {
    return (
      <div className="min-h-screen bg-white flex flex-col items-center justify-start p-6 pb-24 font-sans text-black relative pt-32">
        <ProgressBar step={currentStep} />
        <div className="absolute bottom-4 left-4 text-xs text-gray-500 font-sans uppercase tracking-wider">
          {APP_VER_INFO}
        </div>
        <h1 className="text-[25px] font-medium tracking-tight mb-12">2/4 Initial questionnaire</h1>
        <div className="text-center space-y-8">
          <p className="text-base text-black font-sans">In this phase you should rate the agent’s properties.</p>
          <button onClick={nextStep} className="px-12 py-3 bg-white text-black border border-black rounded-lg text-base font-sans hover:bg-gray-50 transition-all">
            Begin initial questionnaire
          </button>
        </div>
      </div>
    );
  }

  // --- 6. INITIAL QUESTIONNAIRE ---
  if (currentStep === ExperimentStep.INITIAL_QUESTIONNAIRE) {
    const cats = QUESTIONNAIRE_STRUCTURE.filter(c => c.id === "cat1" || c.id === "cat2");
    const allAnswered = cats.flatMap(c => c.questions).every(q => initialQuestionnaireAnswers[q.id]);

    return (
      <div className="min-h-screen bg-white flex flex-col items-center p-6 font-sans text-black relative pt-32">
        <ProgressBar step={currentStep} />
        <div className="absolute bottom-4 left-4 text-xs text-gray-500 font-sans uppercase tracking-wider">
          {APP_VER_INFO}
        </div>
        <h1 className="text-[25px] font-medium mb-12 tracking-tight text-center">2/4 Initial questionnaire</h1>
        <div className="w-full max-w-2xl pb-12 space-y-20">
          {cats.map(category => (
            <section key={category.id} className="space-y-10 group">
              <div className="space-y-4">
                <h2 className="text-base font-bold text-gray-500 border-b border-black pb-2 inline-block tracking-tight">{category.title}</h2>
                <p className="text-base text-black font-sans italic leading-relaxed">{category.description}</p>
              </div>
              <div className="space-y-16">
                {category.questions.map(q => (
                  <div key={q.id} className="space-y-6">
                    <p className="text-base text-black font-sans leading-tight">{q.text}</p>
                    <div className="flex justify-between items-center bg-gray-50 p-5 rounded-2xl border border-dotted border-gray-300">
                      <span className="text-sm text-black font-sans w-32 text-center leading-tight">{q.min}</span>
                      <div className="flex gap-2">
                         {[1, 2, 3, 4, 5, 6, 7].map(num => (
                          <button
                            key={num}
                            onClick={() => setInitialQuestionnaireAnswers(prev => ({ ...prev, [q.id]: num }))}
                            className={`w-12 h-12 rounded-full border border-black flex items-center justify-center text-base font-sans transition-all ${initialQuestionnaireAnswers[q.id] === num ? "bg-gray-200 text-black scale-110 border-2" : "bg-white text-black hover:bg-gray-100"}`}
                          >
                            {num}
                          </button>
                        ))}
                      </div>
                      <span className="text-sm text-black font-sans w-32 text-center leading-tight">{q.max}</span>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          ))}
          <div className="flex justify-center border-t border-black pt-16">
             <button disabled={!allAnswered} onClick={nextStep} className="px-24 py-4 bg-white text-black border border-black rounded-lg text-base font-sans disabled:bg-gray-100 disabled:text-gray-400 transition-all">
               Continue
             </button>
          </div>
        </div>
      </div>
    );
  }

  // --- 7. EXPERIMENT EXPLANATION ---
  if (currentStep === ExperimentStep.EXPERIMENT_EXPLANATION) {
    return (
      <div className="min-h-screen bg-white flex flex-col items-center justify-start p-6 pb-24 font-sans text-black relative pt-32">
        <ProgressBar step={currentStep} />
        <div className="absolute bottom-4 left-4 text-xs text-gray-500 font-sans uppercase tracking-wider">
          {APP_VER_INFO}
        </div>
        <h1 className="text-[25px] font-medium tracking-tight mb-12">3/4 Experiment</h1>
        <div className="text-center space-y-8">
          <div className="space-y-4">
            <p className="text-base text-black font-sans">In this phase you should:</p>
            <ul className="text-left max-w-md mx-auto space-y-1 inline-block list-disc pl-5 text-base text-black font-sans">
              <li>Adjust the loudness to comfortable level</li>
              <li>Engage in a role-play task with the agent (café scenario)</li>
              <li>Input information requested on the screen in the user input field</li>
              <li>Pay attention to the acoustic scene</li>
            </ul>
          </div>
          <button onClick={nextStep} className="px-12 py-3 bg-white text-black border border-black rounded-lg text-base font-sans hover:bg-gray-50 transition-all">
            Begin experiment
          </button>
        </div>
      </div>
    );
  }

  // --- 8. EXPERIMENT PHASE ---
  if (currentStep === ExperimentStep.EXPERIMENT_PHASE) {
    return (
      <div className="min-h-screen bg-white flex flex-col items-center justify-start p-6 pb-24 font-sans text-black gap-12 relative pt-32">
        <ProgressBar step={currentStep} />
        <div className="absolute bottom-4 left-4 text-xs text-gray-500 font-sans uppercase tracking-wider">
          {APP_VER_INFO}
        </div>
        <h1 className="text-[25px] font-medium tracking-tight">3/4 Experiment</h1>
        
        <div className="flex flex-col items-center gap-12 w-full max-w-xl">
          <button 
            onClick={toggleSession} 
            disabled={isConnecting || isCooldown}
            className="px-12 py-4 bg-white border border-black rounded-lg text-base font-sans hover:bg-gray-50 transition-all text-black"
          >
            {isConnecting ? <Loader2 className="animate-spin" /> : isActive ? "Stop session" : "Start session"}
          </button>
          
          <div className="relative w-full aspect-[16/4] border border-black rounded-[24px] overflow-hidden bg-white group select-none">
            <img
              src="https://images.unsplash.com/photo-1554118811-1e0d58224f24?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8N3x8Y2FmZXxlbnwwfHwwfHx8MA%3D%3D"
              alt="Acoustic experiment scene"
              className={`absolute inset-0 w-full h-full object-cover object-[center_75%] transition-all duration-300 ${
                isActive 
                  ? "opacity-60 grayscale-0 brightness-100" 
                  : "opacity-35 grayscale brightness-105"
              }`}
              referrerPolicy="no-referrer"
            />
            {isActive ? (
              <div className="absolute inset-0 bg-white/10 flex flex-col items-center justify-center p-4 z-10">
                <div className="w-48 p-1 bg-white border border-black">
                  <AudioVisualizer session={sessionRef.current} />
                </div>
              </div>
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-xs text-gray-400 font-mono uppercase tracking-widest bg-gray-50/20">
                Audio scene off
              </div>
            )}
          </div>

          <div className="w-full p-10 space-y-8 bg-[#FFFDC0] rounded-[40px] border border-black shadow-sm">
            <div className="grid grid-cols-2 gap-8 font-sans text-xs text-gray-500 uppercase tracking-wider border-b border-black pb-4">
              <div className="font-bold">Information to collect:</div>
              <div className="font-bold">What I understood:</div>
            </div>
            <div className="space-y-6 text-black">
              {tasks.map(task => (
                <div key={task.id} className="grid grid-cols-2 gap-8 border-b border-dotted border-gray-300 pb-2 items-end">
                  <div className="text-base font-sans">{task.text}</div>
                  <input
                    type="text"
                    value={task.understanding}
                    onChange={(e) => updateTask(task.id, "understanding", e.target.value)}
                    className="bg-transparent text-base font-sans text-black focus:outline-none placeholder:text-gray-300 italic"
                    placeholder="Type here..."
                  />
                </div>
              ))}
            </div>
          </div>

          <div className="flex flex-col items-center gap-4 text-black">
            <button 
              onClick={() => {
                if (isActive) {
                  if (window.confirm("Are you sure you want to close the session and continue?")) {
                    nextStep();
                  }
                } else {
                  nextStep();
                }
              }} 
              className="px-24 py-4 bg-white text-black border border-black rounded-lg text-base font-sans hover:bg-gray-50 transition-all"
            >
              Continue
            </button>
            <p className="text-xs text-black font-sans">Ensure the audio session is stopped before continuing</p>
          </div>
        </div>
      </div>
    );
  }

  // --- 9. FINAL QUESTIONNAIRE EXPLANATION ---
  if (currentStep === ExperimentStep.FINAL_QUESTIONNAIRE_EXPLANATION) {
    return (
      <div className="min-h-screen bg-white flex flex-col items-center justify-start p-6 pb-24 font-sans text-black relative pt-32">
        <ProgressBar step={currentStep} />
        <div className="absolute bottom-4 left-4 text-xs text-gray-500 font-sans uppercase tracking-wider">
          {APP_VER_INFO}
        </div>
        <h1 className="text-[25px] font-medium tracking-tight mb-12">4/4 Final questionnaire</h1>
        <div className="text-center space-y-8">
          <p className="text-base text-black font-sans">In this phase you should rate the experiment scenario.</p>
          <button onClick={nextStep} className="px-12 py-3 bg-white text-black border border-black rounded-lg text-base font-sans hover:bg-gray-50 transition-all">
            Begin final questionnaire
          </button>
        </div>
      </div>
    );
  }

  // --- 10. FINAL QUESTIONNAIRE ---
  if (currentStep === ExperimentStep.FINAL_QUESTIONNAIRE) {
    const cats = QUESTIONNAIRE_STRUCTURE.filter(c => (c.id !== "cat1" && c.id !== "cat2"));
    const allAnswered = cats.flatMap(c => c.questions).every(q => !!finalQuestionnaireAnswers[q.id]);

    return (
      <div className="min-h-screen bg-white flex flex-col items-center p-6 font-sans text-black relative pt-32">
        <ProgressBar step={currentStep} />
        <div className="absolute bottom-4 left-4 text-xs text-gray-500 font-sans uppercase tracking-wider">
          {APP_VER_INFO}
        </div>
        <h1 className="text-[25px] font-medium mb-12 tracking-tight text-center">4/4 Final questionnaire</h1>
        <div className="w-full max-w-2xl pb-12 space-y-20 text-black">
          {cats.map(category => (
            <section key={category.id} className="space-y-10 group">
              <div className="space-y-4">
                <h2 className="text-base font-bold text-gray-500 border-b border-black pb-2 inline-block tracking-tight">{category.title}</h2>
                <p className="text-base text-black font-sans italic leading-relaxed">{category.description}</p>
              </div>
              <div className="space-y-16">
                {category.questions.map(q => (
                  <div key={q.id} className="space-y-6">
                    <p className="text-base text-black font-sans leading-tight">{q.text}</p>
                    <div className="flex justify-between items-center bg-gray-50 p-5 rounded-2xl border border-dotted border-gray-300">
                      <span className="text-sm text-black font-sans w-32 text-center leading-tight">{q.min}</span>
                      <div className="flex gap-2">
                        {[1, 2, 3, 4, 5, 6, 7].map(num => (
                          <button
                            key={num}
                            onClick={() => setFinalQuestionnaireAnswers(prev => ({ ...prev, [q.id]: num }))}
                            className={`w-12 h-12 rounded-full border border-black flex items-center justify-center text-base font-sans transition-all ${finalQuestionnaireAnswers[q.id] === num ? "bg-gray-200 text-black scale-110 border-2" : "bg-white text-black hover:bg-gray-100"}`}
                          >
                            {num}
                          </button>
                        ))}
                      </div>
                      <span className="text-sm text-black font-sans w-32 text-center leading-tight">{q.max}</span>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          ))}
          <div className="flex flex-col items-center gap-6 border-t border-black pt-16">
            <button
              disabled={!allAnswered || isUploading}
              onClick={finalizeSession}
              className="px-24 py-4 bg-white text-black border border-black rounded-lg text-base font-sans hover:bg-gray-50 transition-all disabled:bg-gray-100 disabled:text-gray-400"
            >
              {isUploading ? (
                <div className="flex items-center gap-2">
                  <Loader2 className="animate-spin" /> Exporting...
                </div>
              ) : (
                "Finish & export data"
              )}
            </button>
            {!allAnswered && <p className="text-base text-red-500 font-sans">Please complete all fields to finish</p>}
          </div>
        </div>
      </div>
    );
  }

  // --- 11. GOODBYE ---
  return (
    <div className="min-h-screen bg-white flex flex-col items-center justify-center p-6 font-sans text-black relative">
      <div className="absolute bottom-4 left-4 text-xs text-gray-500 font-sans uppercase tracking-wider">
        {APP_VER_INFO}
      </div>
      <div className="w-full max-w-md p-16 text-center">
        <h1 className="text-[25px] font-medium mb-6 font-sans text-center">Session finished</h1>
        <p className="text-black mb-10 leading-relaxed text-base font-sans">
          Thank you for participating! Your data and audio recordings have been exported locally.
        </p>
        <div className="w-20 h-20 rounded-full border border-black flex items-center justify-center mx-auto animate-bounce">
          <Music className="w-8 h-8" size={32} />
        </div>
        <p className="mt-12 text-base text-black italic font-sans">You can close this tab now.</p>
      </div>
    </div>
  );
}
