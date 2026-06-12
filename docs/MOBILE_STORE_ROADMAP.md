# Mobile Store Roadmap — LifeGroups

Status: Draft tracking document  
Owner: Tom  
Product: Fox Valley Church Life Groups  
Primary strategy: Android first, iOS second  
Last updated: 2026-06-12

## 1. Decision

LifeGroups should pursue mobile distribution in this order:

1. **Google Play via PWA + Trusted Web Activity**
2. **Apple App Store via Capacitor iOS shell**
3. **Apple unlisted distribution request after App Review approval**

The app is already a real authenticated ministry operations product, not a static website. That helps App Review viability. The current blocker is packaging and store-readiness, not product legitimacy.

## 2. Current repo baseline

Observed baseline:

- Next.js web app deployed at `https://fvclifegroups.vercel.app/`
- Supabase Auth + Postgres + RLS
- Role-based surfaces: Super Admin, Ministry Admin, Over-Shepherd, Leader
- Mobile-aware CSS exists, including iOS input zoom prevention and mobile stacking helpers
- No PWA manifest found yet
- No service worker/offline strategy found yet
- No Android shell found yet
- No iOS shell found yet
- No store metadata package found yet

## 3. Store strategy

| Platform | Path | Target outcome | Risk |
| --- | --- | --- | --- |
| Google Play | PWA + Trusted Web Activity | Public or limited Play listing | Medium |
| Apple App Store | Capacitor iOS shell | Approved app, preferably unlisted | Medium-high |
| Web | Existing Vercel app | Remains canonical runtime | Low |

## 4. Key review risks

| Risk | Platform | Why it matters | Mitigation |
| --- | --- | --- | --- |
| “Just a website wrapper” | Apple | Apple requires app-like utility beyond a repackaged website | Add native shell polish, launch screen, safe areas, offline/error states, app-specific review notes |
| Login-gated app | Apple + Google | Reviewers need access | Create stable reviewer/demo accounts with fake seeded data |
| Account deletion | Apple + Google | Required if users can create accounts or request account lifecycle changes | Add in-app deletion/request path and public web deletion page |
| Sensitive ministry data | Apple + Google | Notes, prayer requests, and personal info require clear disclosures | Publish privacy policy and align store data forms |
| Mobile UX debt | Apple + Google | Small text, contrast, and long mobile scroll can hurt review and adoption | Fix P0 accessibility/mobile issues before packaging |
| TWA verification failure | Google | Failed Digital Asset Links causes browser UI fallback | Add `.well-known/assetlinks.json` after package/signing is known |

## 5. Phase plan

### Phase 0 — Readiness inventory

Goal: Confirm scope before writing app shell code.

Checklist:

- [ ] Confirm final app name for stores: `LifeGroups`, `FVC LifeGroups`, or `Fox Valley Church Life Groups`
- [ ] Confirm developer account ownership: personal, church, or organization
- [ ] Confirm whether app should be public, unlisted, internal, or invite-only
- [ ] Confirm reviewer demo roles needed
- [ ] Confirm privacy policy owner and support contact
- [ ] Confirm whether account creation is user-initiated, invite-only, or admin-created only
- [ ] Confirm whether account deletion means self-delete, deletion request, or admin-reviewed deletion

Acceptance gate:

- A one-page store positioning note exists in this file or under `docs/store/`.

### Phase 1 — PWA foundation

Goal: Make the web app installable and ready for Android wrapping.

Checklist:

- [ ] Add `app/manifest.ts` or `public/manifest.webmanifest`
- [ ] Add required app icons: 192x192 and 512x512 minimum
- [ ] Add maskable icon
- [ ] Add Apple touch icon
- [ ] Add theme color and background color
- [ ] Add viewport metadata with safe mobile behavior
- [ ] Add app screenshots for documentation/testing
- [ ] Run Lighthouse PWA checks
- [ ] Verify install prompt behavior on Android Chrome
- [ ] Verify login flow works when launched from installed PWA

Suggested manifest values:

```json
{
  "name": "Fox Valley Church Life Groups",
  "short_name": "LifeGroups",
  "start_url": "/",
  "scope": "/",
  "display": "standalone",
  "background_color": "#fbfaf4",
  "theme_color": "#fbfaf4",
  "description": "Ministry operations for Fox Valley Church Life Groups."
}
```

Acceptance gate:

- Android Chrome can install the app and launch it in standalone mode.

### Phase 2 — Mobile UX and review hardening

Goal: Fix the issues most likely to hurt app review or real users.

Checklist:

- [ ] Fix contrast carve-out until WCAG AA passes for normal text and primary buttons
- [ ] Enforce readable mobile type floor
- [ ] Add mobile viewport Playwright project
- [ ] Add iPhone-sized smoke tests for login, Home, Care, Plan, Multiply, Leader surface
- [ ] Add Android-sized smoke tests for the same surfaces
- [ ] Add first-run orientation for Leaders and Over-Shepherds
- [ ] Add network/offline error state that is app-like, not a browser error
- [ ] Confirm keyboard behavior on iOS Safari and Android Chrome
- [ ] Confirm drawer and sticky submit behavior on 375px width

Acceptance gate:

- Mobile smoke tests pass on at least one iPhone-sized viewport and one Android-sized viewport.

### Phase 3 — Privacy, support, and account lifecycle

Goal: Prepare required policy surfaces before store submission.

Checklist:

- [ ] Add public privacy policy page
- [ ] Add public support/contact page
- [ ] Add account deletion or deletion request page
- [ ] Link deletion/support/privacy from authenticated settings or account area
- [ ] Document data categories collected: name, email, role, group data, care notes, prayer requests, audit events, analytics if used
- [ ] Document data sharing: Supabase, Vercel Analytics, Vercel Speed Insights, email provider if used
- [ ] Confirm no unnecessary device permissions are requested
- [ ] Confirm push notifications are not used unless intentionally added later
- [ ] Add reviewer notes explaining limited-audience ministry operations use

Acceptance gate:

- Store data forms can be completed from documented facts, without guessing.

### Phase 4 — Android package via Trusted Web Activity

Goal: Build and test the Google Play package.

Checklist:

- [ ] Install Bubblewrap tooling locally
- [ ] Initialize Android project from production manifest
- [ ] Choose Android package name, likely `org.foxvalleychurch.lifegroups` or similar
- [ ] Generate signing key
- [ ] Build APK/AAB
- [ ] Add Digital Asset Links at `/.well-known/assetlinks.json`
- [ ] Verify TWA opens fullscreen without browser UI
- [ ] Test login/session persistence in TWA
- [ ] Create Google Play app record
- [ ] Complete App access form with reviewer credentials
- [ ] Complete Data safety form
- [ ] Complete account deletion form
- [ ] Upload closed/internal test build
- [ ] Recruit testers if required by account type
- [ ] Run closed test until eligible for production access

Commands sketch:

```bash
npm i -g @bubblewrap/cli
bubblewrap init --manifest=https://fvclifegroups.vercel.app/manifest.webmanifest
bubblewrap build
bubblewrap install
```

Acceptance gate:

- Android test build opens as a verified TWA and passes Play Console review checks.

### Phase 5 — iOS shell via Capacitor

Goal: Build an iOS app shell that does not feel like a lazy website wrapper.

Checklist:

- [ ] Add Capacitor to repo or create `/mobile/ios-shell` workspace
- [ ] Configure app id, app name, icon, splash screen
- [ ] Point shell to production web app or bundled web build, based on chosen architecture
- [ ] Add native launch screen
- [ ] Add safe-area styling checks
- [ ] Add offline/network unavailable screen
- [ ] Test login/session behavior on physical iPhone
- [ ] Test invite/reset/password flows from app context
- [ ] Add App Store screenshots
- [ ] Prepare App Review notes
- [ ] Provide demo account credentials
- [ ] Submit TestFlight build
- [ ] Submit App Store Review
- [ ] After approval, request unlisted app distribution if desired

Acceptance gate:

- iOS build is accepted by TestFlight review and is ready for App Store Review.

### Phase 6 — Store launch

Goal: Release deliberately, with support and rollback clarity.

Checklist:

- [ ] Publish Android internal/closed testing
- [ ] Publish Android production or limited release
- [ ] Publish iOS unlisted or standard listing
- [ ] Send install instructions to intended users
- [ ] Monitor auth errors, crash reports, and user feedback
- [ ] Track first-week issues in GitHub issues
- [ ] Add release notes for mobile users

Acceptance gate:

- Intended church users can install, sign in, and complete their role-specific workflows.

## 6. Reviewer demo account plan

Create fake seeded data only. Do not expose real ministry data to reviewers.

| Role | Needed? | Purpose |
| --- | --- | --- |
| Ministry Admin | Yes | Review Care, Plan, Multiply, Groups, People, Settings |
| Over-Shepherd | Yes | Review coverage-scoped care surface |
| Leader | Yes | Review leader group care/calendar surface |
| Super Admin | Maybe | Avoid unless reviewer needs to see app configuration |

Reviewer notes should explain:

- This is a limited-audience ministry operations app.
- Members do not log in.
- Invited leaders and ministry staff use the app to manage care, follow-ups, group health, and launch planning.
- Demo data is synthetic.
- Sensitive real pastoral content is not included in the review environment.

## 7. Store metadata draft

Working app name options:

1. Fox Valley Church Life Groups
2. FVC LifeGroups
3. LifeGroups by Fox Valley Church

Working short description:

> Ministry operations for Life Group leaders and oversight teams.

Working long description:

> LifeGroups helps Fox Valley Church ministry leaders care for Life Group leaders, track follow-ups, manage group health, plan placements, and identify when new groups may be needed. Access is invite-only and role-based for ministry staff, over-shepherds, and group leaders.

Audience:

- Ministry staff
- Over-Shepherds
- Life Group leaders and co-leaders
- Not intended for public member browsing

Category options:

- Productivity
- Lifestyle
- Organization-specific internal/community tool

## 8. Agent prompts

### Prompt A — PWA readiness

```text
Review this Next.js repo and implement the minimum PWA foundation for Android/iOS installability. Add a web manifest, complete icon references, theme/background colors, Apple touch icon support, and metadata updates. Do not change product behavior. Verify with build, lint, typecheck, and a Lighthouse/PWA-oriented checklist. Output a short summary of changed files and remaining store-readiness gaps.
```

### Prompt B — Mobile UX hardening

```text
Review the LifeGroups app for mobile app-store readiness. Focus only on P0 mobile UX blockers: contrast, tiny text, iPhone/Android viewport behavior, safe sticky actions, login flow, Home, Care, Plan, Multiply, Over-Shepherd, and Leader surfaces. Add Playwright mobile viewport smoke tests where practical. Do not expand scope or redesign the product. Produce a concise pass/fail checklist with fixes made.
```

### Prompt C — Privacy/support/account lifecycle

```text
Add store-readiness policy surfaces for LifeGroups: public privacy policy, support/contact page, and account deletion or deletion-request flow. Keep language accurate to the app's current data model and services. Do not overclaim compliance. Link these surfaces from appropriate public/authenticated areas. Summarize data categories collected, processors used, and any open questions before store submission.
```

### Prompt D — Android TWA package

```text
Prepare this repo for Google Play distribution using Trusted Web Activity. Keep the Vercel web app as the canonical runtime. Use the production PWA manifest, create or document the Android package setup, Digital Asset Links requirements, signing/key implications, and Play Console checklist. Do not ship secrets. Output exact local commands and files that must be committed.
```

### Prompt E — iOS Capacitor shell

```text
Prepare an iOS app-store path for LifeGroups using Capacitor or a minimal native shell. The goal is an app-like wrapper for an authenticated limited-audience ministry operations app, not a rebuild. Add shell configuration, app icon/splash plan, offline/network handling requirements, App Review notes, and TestFlight checklist. Flag anything that could trigger Apple's minimum-functionality rejection.
```

## 9. Definition of done

The mobile-store effort is done when:

- Google Play build is approved and installable by intended testers/users
- iOS build is approved by Apple, preferably unlisted if appropriate
- Privacy/support/deletion pages are live
- Reviewer demo accounts exist with synthetic data
- Mobile smoke tests cover core role-based flows
- Store metadata matches actual product behavior
- No real ministry data is exposed during review

## 10. Do not do yet

Avoid these until the store path proves value:

- Full React Native rebuild
- Push notifications
- Offline data sync
- Native calendar/contact integrations
- Public self-service signup
- Public member-facing app expansion

These may be useful later, but they are not needed for the first store submission.
