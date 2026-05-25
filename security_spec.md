# Security Specification: ChatAid Session Export

## Data Invariants
1. Each completed experiment session record is stored under `/sessions/{sessionId}`.
2. The `sessionId` must be a valid, sanitized alphanumeric string (up to 128 characters) of the format `${alias}_${timestamp}`.
3. Once written, a session document is **immutable** (no updates, no deletions).
4. No client-side reading or listing of sessions is allowed (the database is write-only for clients to ensure participant privacy).
5. All 7 top-level fields defined in `firebase-blueprint.json` are strictly required and no additional fields are permitted.

## The Dirty Dozen (Malicious Payloads)

Here are 12 specific payloads representing attacks against Identity, Integrity, State, and Resource limits that MUST return `PERMISSION_DENIED`:

1. **ID Poisoning (Junk characters):**
   - Attempting to create a session document where `sessionId` contains dangerous character injections like slashes or query selectors.
2. **ID Poisoning (Exceeding max length):**
   - Attempting to write a document with an ID exceeding 128 characters.
3. **Ghost Fields injection (Privilege/Schema escalation):**
   - Providing a shadow field (e.g., `role: "admin"` or `isAdmin: true` or `verified: true`) at the root level of the payload.
4. **Missing Required root field:**
   - Attempting to write a session without the `participant` or `metadata` maps.
5. **Type mismatch (Root level):**
   - Specifying `exportId` as a number or object instead of a string.
6. **Type mismatch (Participant child level):**
   - Specifying `participant.alias` as a map/object/boolean instead of a string.
7. **Type mismatch (Metadata child level):**
   - Specifying `metadata.timestamp` as a list of timestamps instead of a string.
8. **Size Overflow attack:**
   - Injecting a multi-megabyte string into `participant.alias` or `metadata.appVersion` (Denial of Wallet).
9. **Update Hijack (Modify existing session):**
   - Attempting to update or overwrite an existing session document.
10. **Unauthorized Deletion:**
    - Attempting to delete an existing session document from the collection.
11. **Malicious Read/Listing (PII Harvesting):**
    - Attempting to get or list other participant session documents from `/sessions`.
12. **Incomplete Demographic sub-fields:**
    - Writing a participant map but omitting required sub-fields (such as `usingHeadphones` or `isListeningExpert`).

## Security Specification Verification

For safety, the compiled `/firestore.rules` must successfully implement defenses for all "Dirty Dozen" payloads.
