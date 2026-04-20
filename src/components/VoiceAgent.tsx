import React, { useState, useEffect, useRef } from "react";
import { Mic, MicOff, Coffee, Music, Clock, Info, Loader2 } from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { LiveAudioSession, INSTRUCTION_PROMPT, EXPERIMENT_PROMPT } from "../lib/gemini";

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
  const [participantAlias, setParticipantAlias] = useState("");
  const [participantAge, setParticipantAge] = useState("");
  const [gender, setGender] = useState("");
  const [isNativeSpeaker, setIsNativeSpeaker] = useState("");
  const [hearingStatus, setHearingStatus] = useState("");
  const [isListeningExpert, setIsListeningExpert] = useState("");
  const [hasConsented, setHasConsented] = useState(false);
  const [userApiKey, setUserApiKey] = useState("");

  const [tasks, setTasks] = useState([
    { id: 1, text: "Price of coffee with milk", understanding: "" },
    { id: 2, text: "Milk options", understanding: "" },
    { id: 3, text: "Is vegan milk more expensive?", understanding: "" },
    { id: 4, text: "What's the cafe specialty cake?", understanding: "" },
    { id: 5, text: "Name of the wifi", understanding: "" },
    { id: 6, text: "Password of the Wifi", understanding: "" },
    { id: 7, text: "Maximum table duration", understanding: "" },
    { id: 8, text: "Evening event", understanding: "" },
    { id: 9, text: "Artist name", understanding: "" },
    { id: 10, text: "Cafe's closing time", understanding: "" },
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
      setTimeout(() => setIsCooldown(false), 3000);

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
          hrtfType: phase === 'instruction' ? 'NONE' : 'CUSTOM',
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
    // 1. Prepare Answers
    const answersContent = tasks
      .map(task => `${task.text}: ${task.understanding || "No answer provided"}`)
      .join("\n");
    
    // Helper to trigger download
    const downloadFile = (content: string, filename: string) => {
      const blob = new Blob([content], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    };

    const prefix = participantAlias ? `${participantAlias}_` : "";

    // 3. Prepare Info
    const infoContent = `Participant Alias: ${participantAlias}\nAge: ${participantAge}\nGender: ${gender}\nNative English Speaker: ${isNativeSpeaker}\nHearing Status: ${hearingStatus}\nListening Expert: ${isListeningExpert}`;

    // Trigger downloads
    downloadFile(answersContent, `${prefix}answers.txt`);
    downloadFile(infoContent, `${prefix}info.txt`);
    
    // Download audio recording if available
    const recordingBlob = sessionRef.current?.getRecording();
    if (recordingBlob) {
      setTimeout(() => {
        const url = URL.createObjectURL(recordingBlob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${prefix}recording.wav`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }, 200);
    }
  };

  useEffect(() => {
    return () => {
      sessionRef.current?.stop();
    };
  }, []);

  if (!isFormSubmitted) {
    return (
      <div className="min-h-screen bg-white flex flex-col items-center justify-center p-6 font-sans text-black">
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
    <div className="min-h-screen bg-white flex flex-col items-center justify-center p-6 font-sans text-black">
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
          <img 
            src={`${import.meta.env.BASE_URL}images/coffee.png`}
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
            ) : isCooldown ? (
              error?.includes("quota") ? "QUOTA LIMIT..." : "WAIT..."
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
            className="px-12 py-4 bg-black text-white rounded-lg text-lg font-medium hover:bg-gray-800 transition-colors shadow-lg"
          >
            Submit data
          </button>
        </div>
      </div>

      {error && (
        <p className="mt-6 text-red-500 text-sm font-medium">{error}</p>
      )}
    </div>
  );
}
