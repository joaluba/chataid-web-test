import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";
import { getAuth, signInAnonymously } from "firebase/auth";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyDKc8fFb13X65O2frNvmgypsp-3Ru85VYg",
  authDomain: "chataid-6d36d.firebaseapp.com",
  projectId: "chataid-6d36d",
  storageBucket: "chataid-6d36d.firebasestorage.app",
  messagingSenderId: "427781867551",
  appId: "1:427781867551:web:97f759b781bf948a8e2d2a"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
export const storage = getStorage(app);
export const auth = getAuth(app);

export const loginAnonymously = async () => {
    try {
        await signInAnonymously(auth);
    } catch (e) {
        console.error("Auth failed:", e);
    }
};
