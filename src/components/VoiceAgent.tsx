import React, { useState, useEffect, useRef } from "react";
import { Mic, MicOff, Coffee, Music, Clock, Info, Loader2 } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { LiveAudioSession, INSTRUCTION_PROMPT, EXPERIMENT_PROMPT } from "../lib/gemini";
import { db, loginAnonymously } from "../lib/firebase";
import { collection, addDoc, serverTimestamp } from "firebase/firestore";
import JSZip from "jszip";

const APP_VER_INFO = `App version: v0.0.2 (First Questionnaire) SNR=${LiveAudioSession.SNR_DB}`;

const QUESTIONNAIRE_STRUCTURE = [
  {
    id: "cat1",
    title: "Category 1: TTS Voice Quality",
    description: "Rate the quality of the generated voice itself, as independently as possible from the café noise or background sounds. Focus on the voice’s clarity, naturalness, pleasantness, rhythm, intonation, and speaking style, not on how difficult or realistic the acoustic scene was.",
    questions: [
      { id: "effort", text: "Please rate the degree of effort you had to make to understand the message.", min: "1 = impossible even with much effort", max: "7 = no effort required" },
      { id: "singleWords", text: "Were single words hard to understand?", min: "1 = all words hard to understand", max: "7 = all words easy to understand" },
      { id: "clearlyDistinguishable", text: "Were the speech sounds clearly distinguishable?", min: "1 = not at all clear", max: "7 = very clear" },
      { id: "preciseArticulation", text: "Was the articulation of speech sounds precise?", min: "1 = slurred or imprecise", max: "7 = precise" },
      { id: "pleasantVoice", text: "Was the voice you heard pleasant to listen to?", min: "1 = very unpleasant", max: "7 = very pleasant" },
      { id: "naturalVoice", text: "Did the voice sound natural?", min: "1 = very unnatural", max: "7 = very natural" },
      { id: "humanLike", text: "To what extent did the voice sound like a human?", min: "1 = nothing like a human", max: "7 = just like a human" },
      { id: "harshVoice", text: "Did the voice sound harsh, raspy, or strained?", min: "1 = significantly harsh/raspy", max: "7 = normal quality" },
      { id: "emphasis", text: "Did emphasis of important words occur?", min: "1 = incorrect emphasis", max: "7 = excellent use of emphasis" },
      { id: "naturalRhythm", text: "Did the rhythm of the speech sound natural?", min: "1 = unnatural or mechanical", max: "7 = natural rhythm" },
      { id: "smoothIntonation", text: "Did the intonation pattern of sentences sound smooth and natural?", min: "1 = abrupt or abnormal", max: "7 = smooth or normal" },
      { id: "trustworthy", text: "Did the voice appear to be trustworthy?", min: "1 = not at all trustworthy", max: "7 = very trustworthy" },
      { id: "confident", text: "Did the voice suggest a confident speaker?", min: "1 = not at all confident", max: "7 = very confident" },
      { id: "enthusiastic", text: "Did the voice seem to be enthusiastic?", min: "1 = not at all enthusiastic", max: "7 = very enthusiastic" },
      { id: "persuasive", text: "Was the voice persuasive?", min: "1 = not at all persuasive", max: "7 = very persuasive" },
    ]
  },
  {
    id: "cat2",
    title: "Category 2: Agent Interaction Quality",
    description: "Rate the quality of the interaction with the voice agent, focusing on timing, turn-taking, delays, pauses, interruptions, and conversational flow. Do not rate the background noise itself, except when it directly affected how smoothly the interaction worked.",
    questions: [
      { id: "delayAcceptable", text: "The delay between my speech and the agent's reply felt acceptable.", min: "1 = strongly disagree", max: "7 = strongly agree" },
      { id: "freeFlowing", text: "How free-flowing did the conversation feel?", min: "1 = not at all free-flowing", max: "7 = completely free-flowing" },
      { id: "naturalConv", text: "How natural did the conversation feel?", min: "1 = not at all natural", max: "7 = completely natural" },
      { id: "awkwardPauses", text: "How often during the conversation did you experience awkward pauses, interruptions, or talking over the agent?", min: "1 = never", max: "7 = very frequently" },
    ]
  },
  {
    id: "cat3",
    title: "Category 3: Passive Listening Difficulty",
    description: "Rate how difficult it was to understand the voice agent while listening in the noisy café-like acoustic scene. Here, you should take the background noise and overall listening conditions into account, focusing on how much you understood and how much effort listening required.",
    questions: [
      { id: "understandAmount", text: "How much of the agent's speech could you understand in this conversation?", min: "1 = not at all", max: "7 = everything" },
      { id: "understandEffort", text: "How much effort did it take you to understand the agent's speech?", min: "1 = no effort at all", max: "7 = extreme effort" },
    ]
  },
  {
    id: "cat4",
    title: "Category 4: Communication Difficulty",
    description: "Rate how difficult it was to communicate (interact) with the agent in the noisy café-like acoustic scene. Take the acoustic scene into account when judging stress, repetition, smooth back-and-forth exchange, and overall communication success.",
    questions: [
      { id: "stressful", text: "How stressful was it to have this conversation?", min: "1 = not stressful at all", max: "7 = extremely stressful" },
      { id: "noiseDifficulty", text: "The noise made it hard to have a smooth back-and-forth exchange with the agent.", min: "1 = strongly disagree", max: "7 = strongly agree" },
      { id: "commSuccess", text: "How successful was the communication overall?", min: "1 = completely unsuccessful", max: "7 = completely successful" },
      { id: "askRepeat", text: "How often did you have to ask the agent to repeat or rephrase something?", min: "1 = never", max: "7 = very frequently" },
    ]
  },
  {
    id: "cat5",
    title: "Category 5: Task Ecological Validity",
    description: "Rate how realistic and meaningful the role-play task felt as a situation you might encounter in everyday life. Focus on the task and your engagement with it, not on the voice quality or background noise.",
    questions: [
      { id: "relevance", text: "How relevant was this listening situation to your everyday life?", min: "1 = not at all relevant", max: "7 = extremely relevant" },
      { id: "imagineDoing", text: "The task I was asked to complete is something I could imagine doing in a real café.", min: "1 = strongly disagree", max: "7 = strongly agree" },
      { id: "engaged", text: "How engaged did you feel in the role-play task?", min: "1 = not at all engaged", max: "7 = extremely engaged" },
    ]
  },
  {
    id: "cat6",
    title: "Category 6: Acoustic Scene Ecological Validity",
    description: "Rate how realistic and immersive the café-like acoustic environment sounded. Focus on the background noise, the sense of being in a real café, and whether the target voice fit naturally into that environment.",
    questions: [
      { id: "realCafe", text: "The acoustic environment sounded like a real café I might find myself in.", min: "1 = strongly disagree", max: "7 = strongly agree" },
      { id: "immersive", text: "How immersive did the acoustic environment feel?", min: "1 = not at all immersive", max: "7 = extremely immersive" },
      { id: "realisticNoise", text: "How realistic was the background noise for a café environment?", min: "1 = not at all realistic", max: "7 = completely realistic" },
      { id: "realisticVoice", text: "How realistic was the target voice sound for a café environment?", min: "1 = not at all realistic", max: "7 = completely realistic" },
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

export default function VoiceAgent() {
  const [isActive, setIsActive] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isCooldown, setIsCooldown] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasEnded, setHasEnded] = useState(false);
  const [phase, setPhase] = useState<'instruction' | 'experiment'>('instruction');
  const sessionRef = useRef<LiveAudioSession | null>(null);

  // User data form state
  const [isFormSubmitted, setIsFormSubmitted] = useState(false);
  const [isQuestionnaireActive, setIsQuestionnaireActive] = useState(false);
  const [isFinished, setIsFinished] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isAuthorized, setIsAuthorized] = useState(false);
  const [passphrase, setPassphrase] = useState("");
  const [passphraseError, setPassphraseError] = useState(false);
  const [questionnaireAnswers, setQuestionnaireAnswers] = useState<Record<string, number>>(() => {
    const initial: Record<string, number> = {};
    QUESTIONNAIRE_STRUCTURE.forEach(cat => {
      cat.questions.forEach(q => {
        initial[q.id] = 4;
      });
    });
    return initial;
  });
  
  const [participantAlias, setParticipantAlias] = useState("Tester");
  const [participantAge, setParticipantAge] = useState("25");
  const [gender, setGender] = useState("Male");
  const [isNativeSpeaker, setIsNativeSpeaker] = useState("yes");
  const [hearingStatus, setHearingStatus] = useState("Normal hearing");
  const [isListeningExpert, setIsListeningExpert] = useState("no");
  const [hasConsented, setHasConsented] = useState(true);
  const [userApiKey, setUserApiKey] = useState("");

  const [tasks, setTasks] = useState([
    { id: 1, text: "Price of coffee with milk", understanding: "Test answer" },
    { id: 2, text: "Milk options", understanding: "Test answer" },
    { id: 3, text: "Is vegan milk more expensive?", understanding: "Test answer" },
    { id: 4, text: "What's the cafe specialty cake?", understanding: "Test answer" },
    { id: 5, text: "Name of the wifi", understanding: "Test answer" },
    { id: 6, text: "Password of the Wifi", understanding: "Test answer" },
    { id: 7, text: "Maximum table duration", understanding: "Test answer" },
    { id: 8, text: "Evening event", understanding: "Test answer" },
    { id: 9, text: "Artist name", understanding: "Test answer" },
    { id: 10, text: "Cafe's closing time", understanding: "Test answer" },
  ]);

  const updateTask = (id: number, field: "understanding", value: string) => {
    setTasks(prev => prev.map(t => t.id === id ? { ...t, [field]: value } : t));
  };

  const toggleSession = async () => {
    if (isCooldown) return;

    if (isActive) {
      sessionRef.current?.stop();
      setIsActive(false);
      
      // Add a small cooldown after stopping to prevent rapid restarts
      setIsCooldown(true);
      setTimeout(() => setIsCooldown(false), 1000);

      if (phase === 'instruction') {
        setPhase('experiment');
      } else {
        setHasEnded(true);
      }
    } else {
      setIsConnecting(true);
      setError(null);
      setHasEnded(false);
      try {
        if (!sessionRef.current) {
          const apiKey = userApiKey || process.env.GEMINI_API_KEY;
          if (!apiKey) {
            throw new Error("No Gemini API key provided. Please enter one in the form or configure a default key.");
          }
          sessionRef.current = new LiveAudioSession(apiKey);
        }
        await sessionRef.current.start({
          systemInstruction: phase === 'instruction' ? INSTRUCTION_PROMPT : EXPERIMENT_PROMPT,
          shouldPlayNoise: phase === 'experiment',
          onError: (err) => {
            const errorMessage = err?.message || String(err);
            console.log(err)
            if (errorMessage.toLowerCase().includes("quota")) {
              setError("Gemini API quota reached. The Free Tier limit is usually 3-5 sessions per minute. Please wait exactly 60 seconds for the limit to reset.");
              setIsCooldown(true);
              setTimeout(() => setIsCooldown(false), 10000); // 10s lockout for quota errors
            } else {
              setError("Something went wrong with the connection.");
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
        const errorMessage = err?.message || String(err);
        if (errorMessage.toLowerCase().includes("quota")) {
          setError("Gemini API quota exceeded. Please wait 60 seconds for a full reset.");
          setIsCooldown(true);
          setTimeout(() => setIsCooldown(false), 10000);
        } else {
          setError("Could not access microphone or connect to Ramona.");
        }
      } finally {
        setIsConnecting(false);
      }
    }
  };

  const handleSubmit = () => {
    setIsQuestionnaireActive(true);
  };

  const finalizeSession = async () => {
    setIsUploading(true);
    setError(null);

    // 1. Prepare Full Data Dictionary
    const fullDataDictionary = {
      participant: {
        alias: participantAlias,
        age: participantAge,
        gender,
        isNativeSpeaker,
        hearingStatus,
        isListeningExpert
      },
      tasks: tasks.reduce((acc, t) => ({ ...acc, [t.text]: t.understanding }), {}),
      questionnaire: questionnaireAnswers,
      metadata: {
        appVersion: APP_VER_INFO,
        timestamp: new Date().toISOString()
      }
    };

    try {
      // 2. Auth for security
      await loginAnonymously();

      // 3. Upload to Firestore (Text Data)
      let docId = "local_backup";
      try {
        const docRef = await addDoc(collection(db, "results"), {
          ...fullDataDictionary,
          serverTimestamp: serverTimestamp()
        });
        docId = docRef.id;
      } catch (firestoreErr) {
        console.error("Firestore upload failed:", firestoreErr);
        // We'll continue even if Firestore fails
      }

      // 4. Create ZIP and trigger download
      const zip = new JSZip();
      
      // Add JSON data
      zip.file(`${docId}_data.json`, JSON.stringify(fullDataDictionary, null, 2));

      // Add audio recordings if available
      const recordings = sessionRef.current?.getRecordings();
      if (recordings) {
        if (recordings.transcript) zip.file(`${docId}_transcript.wav`, recordings.transcript);
        if (recordings.voice) zip.file(`${docId}_voice.wav`, recordings.voice);
        if (recordings.noise) zip.file(`${docId}_noise.wav`, recordings.noise);
      }

      // Generate and download
      const content = await zip.generateAsync({ type: "blob" });
      const url = URL.createObjectURL(content);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${docId}_${participantAlias || "anonymous"}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      setIsFinished(true);
      setIsQuestionnaireActive(false);
    } catch (err: any) {
      console.error("Critical finalize error:", err);
      setError("An error occurred during export. Your data has been recorded in our database, but the local download might have failed.");
    } finally {
      setIsUploading(false);
    }
  };

  useEffect(() => {
    return () => {
      sessionRef.current?.stop();
    };
  }, []);

  if (!isAuthorized) {
    const handleAuthorize = () => {
      // Define your passphrase here
      if (passphrase === "ChatAid2026") {
        setIsAuthorized(true);
      } else {
        setPassphraseError(true);
      }
    };

    return (
      <div className="min-h-screen bg-white flex flex-col items-center justify-center p-6 font-sans text-black">
        <div className="w-full max-w-sm border border-black rounded-[40px] p-10 text-center space-y-8">
          <h1 className="text-2xl font-medium">Restricted Access</h1>
          <p className="text-gray-500 text-sm italic leading-relaxed">
            Please enter the passphrase provided by the researcher to continue.
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
              className={`w-full text-center text-sm bg-transparent border-b ${passphraseError ? "border-red-500" : "border-black"} pb-1 focus:outline-none`}
              placeholder="Enter passphrase"
              autoFocus
            />
            {passphraseError && (
              <p className="text-[10px] text-red-500 italic">Incorrect passphrase. Please try again.</p>
            )}
          </div>
          <button
            onClick={handleAuthorize}
            className="w-full py-3 bg-black text-white rounded-lg text-lg font-medium hover:bg-gray-800 transition-colors shadow-lg"
          >
            Enter
          </button>
        </div>
      </div>
    );
  }

  if (isFinished) {
    return (
      <div className="min-h-screen bg-white flex flex-col items-center justify-center p-6 font-sans text-black relative">
        <div className="absolute bottom-4 left-4 text-[10px] text-gray-400 uppercase tracking-widest font-medium">
          {APP_VER_INFO}
        </div>
        <div className="w-full max-w-md border border-black rounded-[40px] p-8 md:p-10 my-8">
          <h1 className="text-2xl font-medium mb-6 text-center">Session Finished</h1>
          <p className="text-gray-600 text-center mb-8 leading-relaxed">
            Thank you for participating! Your data and audio recordings have been exported.
          </p>
          <div className="w-20 h-20 rounded-full border border-black flex items-center justify-center mx-auto">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 200, damping: 10 }}
            >
              <Music className="w-8 h-8" />
            </motion.div>
          </div>
        </div>
      </div>
    );
  }

  if (isQuestionnaireActive) {
    const allQuestions = QUESTIONNAIRE_STRUCTURE.flatMap(cat => cat.questions);
    const allQuestionsAnswered = allQuestions.every(q => questionnaireAnswers[q.id]);

    return (
      <div className="min-h-screen bg-white flex flex-col items-center p-6 font-sans text-black relative">
        <div className="absolute bottom-4 left-4 text-[10px] text-gray-400 uppercase tracking-widest font-medium">
          {APP_VER_INFO}
        </div>
        
        <div className="w-full max-w-2xl py-12">
          <h1 className="text-3xl font-medium text-center mb-8">Questionnaire</h1>
          <div className="text-center text-black mb-12 max-w-xl mx-auto space-y-2">
            <p className="leading-relaxed text-sm">
              Please rate each statement using the 7-point scale shown next to it. 
              Some sections ask you to ignore the <b>background acoustic scene</b> and focus only on the agent or interaction, while other sections ask you to take the noisy café-like environment into account.
              There are no right or wrong answers; we are interested in your own listening and interaction experience!
            </p>
          </div>

          <div className="space-y-24">
            {QUESTIONNAIRE_STRUCTURE.map((category) => (
              <section key={category.id} className="space-y-10 group">
                <div className="space-y-4">
                  <h2 className="text-xl font-medium border-b border-black pb-2 inline-block">
                    {category.title}
                  </h2>
                  <p className="text-sm text-gray-600 leading-relaxed italic pr-8">
                    {category.description}
                  </p>
                </div>

                <div className="space-y-16">
                  {category.questions.map((q) => (
                    <div key={q.id} className="space-y-6">
                      <p className="text-sm font-medium leading-tight">{q.text}</p>
                      <div className="flex flex-col gap-2">
                        <div className="flex justify-between items-center bg-gray-50 p-4 sm:p-5 rounded-2xl border border-dotted border-gray-300">
                          <span className="text-[10px] text-black w-24 text-center leading-tight">{q.min}</span>
                          <div className="flex gap-2 sm:gap-4 px-2">
                            {[1, 2, 3, 4, 5, 6, 7].map((num) => (
                              <button
                                key={num}
                                onClick={() => setQuestionnaireAnswers(prev => ({ ...prev, [q.id]: num }))}
                                className={`w-8 h-8 sm:w-11 sm:h-11 rounded-full border border-black flex items-center justify-center text-xs transition-all ${
                                  questionnaireAnswers[q.id] === num 
                                    ? "bg-black text-white scale-110 shadow-lg" 
                                    : "bg-white hover:bg-gray-100 hover:scale-105"
                                }`}
                              >
                                {num}
                              </button>
                            ))}
                          </div>
                          <span className="text-[10px] text-black w-24 text-center leading-tight">{q.max}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            ))}
          </div>

          <div className="mt-24 flex flex-col items-center gap-6 border-t border-black pt-16 uppercase tracking-tight">
            {!allQuestionsAnswered && (
              <p className="text-xs text-red-400 italic animate-pulse">Please answer all questions to export your data.</p>
            )}
            
            <div className="flex flex-col sm:flex-row gap-4 w-full justify-center">
              <button
                onClick={finalizeSession}
                disabled={!allQuestionsAnswered || isUploading}
                className="px-12 py-3 bg-black text-white rounded-lg text-lg font-medium hover:bg-gray-800 transition-all shadow-lg disabled:bg-gray-200 disabled:text-gray-400 disabled:border-transparent disabled:cursor-not-allowed flex items-center justify-center gap-3"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="animate-spin w-5 h-5" />
                    Exporting...
                  </>
                ) : (
                  "Finish & Export Data"
                )}
              </button>
            </div>

            {error && (
              <div className="text-center p-4 bg-red-50 border border-red-100 rounded-xl max-w-lg">
                <p className="text-xs text-red-600">{error}</p>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (!isFormSubmitted) {
    return (
      <div className="min-h-screen bg-white flex flex-col items-center justify-center p-6 font-sans text-black relative">
        <div className="absolute bottom-4 left-4 text-[10px] text-gray-400 uppercase tracking-widest font-medium">
          {APP_VER_INFO}
        </div>
        <div className="w-full max-w-md border border-black rounded-[40px] p-8 md:p-10 my-8">
          <h1 className="text-2xl font-medium mb-8 text-center">Participant Information</h1>
          <div className="space-y-6">
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">Participant Alias</label>
              <input
                type="text"
                value={participantAlias}
                onChange={(e) => setParticipantAlias(e.target.value)}
                placeholder="Enter your alias"
                className="w-full text-sm bg-transparent border-b border-dotted border-black pb-1 focus:outline-none"
              />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">Age (Optional)</label>
              <input
                type="number"
                value={participantAge}
                onChange={(e) => setParticipantAge(e.target.value)}
                placeholder="Enter your age"
                className="w-full text-sm bg-transparent border-b border-dotted border-black pb-1 focus:outline-none"
              />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">Gender (Optional)</label>
              <select
                value={gender}
                onChange={(e) => setGender(e.target.value)}
                className="w-full text-sm bg-transparent border-b border-dotted border-black pb-1 focus:outline-none"
              >
                <option value="">Select gender</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Non-binary">Non-binary</option>
                <option value="Prefer to self-describe">Prefer to self-describe</option>
              </select>
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">Native English speaker (Optional)</label>
              <div className="flex gap-4">
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <input type="radio" name="nativeSpeaker" value="yes" checked={isNativeSpeaker === "yes"} onChange={(e) => setIsNativeSpeaker(e.target.value)} className="accent-black" /> Yes
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <input type="radio" name="nativeSpeaker" value="no" checked={isNativeSpeaker === "no"} onChange={(e) => setIsNativeSpeaker(e.target.value)} className="accent-black" /> No
                </label>
              </div>
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">Hearing status (Optional)</label>
              <select
                value={hearingStatus}
                onChange={(e) => setHearingStatus(e.target.value)}
                className="w-full text-sm bg-transparent border-b border-dotted border-black pb-1 focus:outline-none"
              >
                <option value="">Select status</option>
                <option value="Normal hearing">Normal hearing</option>
                <option value="Hearing impaired">Hearing impaired</option>
                <option value="Not sure">Not sure</option>
              </select>
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">Listening expert (Optional)</label>
              <div className="flex gap-4">
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <input type="radio" name="listeningExpert" value="yes" checked={isListeningExpert === "yes"} onChange={(e) => setIsListeningExpert(e.target.value)} className="accent-black" /> Yes
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <input type="radio" name="listeningExpert" value="no" checked={isListeningExpert === "no"} onChange={(e) => setIsListeningExpert(e.target.value)} className="accent-black" /> No
                </label>
              </div>
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-sm font-medium">Gemini API Key (Optional)</label>
              <input
                type="password"
                value={userApiKey}
                onChange={(e) => setUserApiKey(e.target.value)}
                placeholder="Leave blank to use default key"
                className="w-full text-sm bg-transparent border-b border-dotted border-black pb-1 focus:outline-none"
              />
            </div>
          </div>

          <div className="mt-8 flex items-start gap-3">
            <input
              type="checkbox"
              id="consent"
              checked={hasConsented}
              onChange={(e) => setHasConsented(e.target.checked)}
              className="mt-1 accent-black cursor-pointer"
            />
            <label htmlFor="consent" className="text-[10px] leading-tight text-gray-600 cursor-pointer">
              I confirm that I have read and understood the information provided regarding this study. I voluntarily agree to participate and provide my informed consent. I understand that the data collected during the study will be anonymized and may be used for research purposes.
            </label>
          </div>

          <div className="mt-10 flex justify-center">
            <button
              onClick={() => setIsFormSubmitted(true)}
              disabled={!participantAlias || !hasConsented}
              className="px-12 py-3 bg-black text-white rounded-lg text-lg font-medium hover:bg-gray-800 transition-colors shadow-lg disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              Continue
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white flex flex-col items-center justify-center p-6 font-sans text-black relative">
      <div className="absolute bottom-4 left-4 text-[10px] text-gray-400 uppercase tracking-widest font-medium">
        {APP_VER_INFO}
      </div>
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-2xl font-medium mb-1">
          {phase === 'instruction' ? "Study Instructions" : "Speech communication test"}
        </h1>
        <p className="text-xl font-medium">
          {phase === 'instruction' ? "Phase 1: Preparation" : "Scenario: Cafe"}
        </p>
      </div>

      {/* Main Interaction Area (The "Image/Gradient" Panel) */}
      <div className="w-full max-w-xl mb-8">
        <div className="w-full aspect-[3/1] rounded-sm border border-black flex items-center justify-center relative overflow-hidden">
          <img src="https://res.cloudinary.com/dqttqwfib/image/upload/f_auto,q_auto/coffee_hkkblh"
            alt="Cafe Vinyl" 
            className="absolute inset-0 w-full h-full object-cover opacity-80"
            referrerPolicy="no-referrer"
          />
          <div className="absolute inset-0 bg-gradient-to-b from-black/20 via-transparent to-black/20" />

          <button
            onClick={toggleSession}
            disabled={isConnecting || isCooldown}
            className="px-8 py-3 bg-white border border-black rounded-lg text-lg font-medium hover:bg-gray-50 transition-colors shadow-sm z-10 min-w-[124px] disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isConnecting ? (
              <Loader2 className="animate-spin mx-auto" size={24} />
            ) : isActive ? (
              phase === 'instruction' ? "STOP INSTRUCTION" : "STOP EXPERIMENT"
            ) : (
              phase === 'instruction' ? "START INSTRUCTION" : "START EXPERIMENT"
            )}
          </button>
        </div>

        {isActive && (
          <div className="mt-4 flex justify-center">
            <div className="w-48 border border-black p-1 bg-white">
              <AudioVisualizer session={sessionRef.current} />
            </div>
          </div>
        )}

        {/* Subtle Transcription Overlay removed as per request */}
      </div>

      {/* Information to Collect Section */}
      <div className="w-full max-w-xl">
        <div className="border border-black rounded-[40px] p-8 md:p-10">
          <div className="grid grid-cols-2 gap-8 mb-6">
            <h2 className="text-lg font-medium">Information to collect:</h2>
            <h2 className="text-lg font-medium">What I understood:</h2>
          </div>
          
          <div className="space-y-6">
            {tasks.map((task) => (
              <div key={task.id} className="grid grid-cols-2 gap-8 items-end">
                <div className="text-sm border-b border-dotted border-black pb-1 min-h-[24px]">
                  {task.text}
                </div>
                <div className="relative">
                  <input
                    type="text"
                    value={task.understanding}
                    onChange={(e) => updateTask(task.id, "understanding", e.target.value)}
                    className="w-full text-sm bg-transparent border-b border-dotted border-black pb-1 focus:outline-none"
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-12 flex justify-center">
          <button
            onClick={handleSubmit}
            className="px-12 py-3 bg-black text-white rounded-lg text-lg font-medium hover:bg-gray-800 transition-colors shadow-lg"
          >
            Continue
          </button>
        </div>
      </div>

      {error && (
        <p className="mt-6 text-red-500 text-sm font-medium">{error}</p>
      )}
    </div>
  );
}
