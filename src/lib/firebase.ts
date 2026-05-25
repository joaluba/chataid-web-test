import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";
import { getAuth, signInAnonymously } from "firebase/auth";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID
};

// Check if configuration is present
const isConfigValid = !!firebaseConfig.apiKey;

if (!isConfigValid) {
  console.warn("Firebase API Key is missing. Please configure environment variables in AI Studio settings.");
}

// Initialize Firebase (safely)
const app = isConfigValid ? initializeApp(firebaseConfig) : null;
export const db = app ? getFirestore(app) : null as any;
export const storage = app ? getStorage(app) : null as any;
export const auth = app ? getAuth(app) : null as any;

export const loginAnonymously = async () => {
    if (!auth) {
        console.error("Firebase Auth not initialized. Check your environment variables.");
        return;
    }
    try {
        await signInAnonymously(auth);
    } catch (e) {
        console.error("Auth failed:", e);
    }
};
