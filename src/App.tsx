/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import VoiceAgent from "./components/VoiceAgent";
import { Analytics } from "@vercel/analytics/react";

export default function App() {
  return (
    <main>
      <VoiceAgent />
      <Analytics />
    </main>
  );
}
