import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";
import { getAuth, signInAnonymously } from "firebase/auth";
import firebaseConfig from "../../firebase-applet-config.json";

// Check if configuration is present
export const isConfigValid = !!firebaseConfig.apiKey;

if (!isConfigValid) {
  console.warn("Firebase API Key is missing from configuration.");
}

// Initialize Firebase with the explicit credentials from the config file
const app = isConfigValid ? initializeApp(firebaseConfig) : null;
export const db = app
  ? (("firestoreDatabaseId" in firebaseConfig && firebaseConfig.firestoreDatabaseId)
      ? getFirestore(app, (firebaseConfig as any).firestoreDatabaseId)
      : getFirestore(app))
  : null as any;
export const storage = app ? getStorage(app) : null as any;
export const auth = app ? getAuth(app) : null as any;

export const loginAnonymously = async () => {
    if (!auth) {
        throw new Error("Firebase Auth not initialized. Check your environment variables.");
    }
    try {
        await signInAnonymously(auth);
    } catch (e) {
        console.error("Auth failed:", e);
        throw e;
    }
};
