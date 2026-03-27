# RouteIQ Tomorrow Execution Plan

Date target: 2026-03-29

This checklist is ordered. Follow it top to bottom.

## 0. Goal

By the end of this plan, you will have:
- Portfolio-ready screenshots in `assets/`
- README updated with those screenshots
- A validated final run checklist
- A GitHub release tag (`v1.0.0`)
- Optional public deployment notes completed

---

## 1. Start Session and Verify Clean State

1. Open terminal and run:

```bash
cd /Users/sauravsz/The-Hub/RouteIQ
source /Users/sauravsz/The-Hub/.venv/bin/activate
git status -sb
```

2. Expected result:
- branch shows `main...origin/main`
- no modified or untracked files

If not clean, stop and resolve before continuing.

---

## 2. Launch App and Capture Screenshots

1. Start app:

```bash
cd /Users/sauravsz/The-Hub/RouteIQ
source /Users/sauravsz/The-Hub/.venv/bin/activate
./scripts/switch_provider.sh google
streamlit run app.py
```

2. Open browser at local Streamlit URL.

3. Capture and save these screenshots in `assets/` with exact names:
- `assets/baseline-dashboard.png`
- `assets/disruption-dashboard.png`
- `assets/cost-surge-dashboard.png`

4. Screenshot rules:
- Include sidebar controls and key metrics in view.
- Keep consistent browser width.
- Prefer PNG format.
- Do not include sensitive tokens or terminal history with keys.

5. Stop Streamlit after screenshots:
- Press `Ctrl + C` in terminal.

---

## 3. Update README Screenshot Section

1. Open `README.md`.
2. Find the `## Screenshots` section.
3. Replace section content with:

```markdown
## Screenshots

### Baseline Scenario
![Baseline Dashboard](assets/baseline-dashboard.png)

### Disruption Scenario
![Disruption Dashboard](assets/disruption-dashboard.png)

### Cost Surge Scenario
![Cost Surge Dashboard](assets/cost-surge-dashboard.png)
```

4. Save file.

---

## 4. Final Smoke Test (Required)

1. Run app again:

```bash
cd /Users/sauravsz/The-Hub/RouteIQ
source /Users/sauravsz/The-Hub/.venv/bin/activate
streamlit run app.py
```

2. Validate this checklist manually:
- Scenario dropdown works (`baseline`, `disruption`, `cost_surge`).
- Cost multiplier slider changes total transportation cost.
- Network flow chart updates by scenario.
- Cost heatmap updates by scenario.
- AI Executive Briefing section updates and returns text.

3. Stop app with `Ctrl + C`.

---

## 5. Commit Screenshot and README Updates

1. Stage, commit, and push:

```bash
cd /Users/sauravsz/The-Hub/RouteIQ
git add README.md assets/baseline-dashboard.png assets/disruption-dashboard.png assets/cost-surge-dashboard.png
git commit -m "Add final dashboard screenshots to README"
git push origin main
```

2. Verify:

```bash
git status -sb
git log --oneline --decorate -n 3
```

Expected:
- clean status
- latest commit visible on `origin/main`

---

## 6. Create Release Tag (v1.0.0)

1. Create and push tag:

```bash
cd /Users/sauravsz/The-Hub/RouteIQ
git tag -a v1.0.0 -m "RouteIQ v1.0.0: phases 1-7 complete"
git push origin v1.0.0
```

2. On GitHub:
- Open repo tags/releases
- Confirm `v1.0.0` exists
- Create release notes (brief)

Release note template:
- Completed Phases 1-7
- Streamlit interactive app
- Multi-scenario optimization
- AI executive briefing

---

## 7. Optional Deployment (If Time Allows)

### Option A: Streamlit Community Cloud

1. Ensure repo is public and updated.
2. In Streamlit Cloud, create new app from this repo.
3. Set main file path to `app.py`.
4. Add environment variables in app settings:
- `AI_PROVIDER`
- provider API key (`GOOGLE_API_KEY` or chosen provider)
- model/base URL variables if needed

5. Deploy and test live URL.

### Option B: Keep Local Only

- Skip deployment and keep GitHub-only portfolio artifact.

---

## 8. End-of-Day Completion Checklist

Mark all done before closing:
- [ ] Screenshots captured in `assets/`
- [ ] README screenshot section updated
- [ ] Final smoke test passed
- [ ] Screenshot commit pushed to `main`
- [ ] `v1.0.0` tag created and pushed
- [ ] Release page updated on GitHub

---

## 9. Troubleshooting Quick Fixes

1. `streamlit: command not found`

```bash
source /Users/sauravsz/The-Hub/.venv/bin/activate
pip install streamlit
```

2. AI briefing unavailable:
- Confirm `.env` exists
- Confirm `AI_PROVIDER` matches available key
- Run quick switch:

```bash
./scripts/switch_provider.sh google
```

3. Git says nothing to commit:
- confirm screenshot files are in `assets/`
- run `git status --short`

---

## 10. Exact Minimal Command Sequence

If you want only the essential command flow tomorrow:

```bash
cd /Users/sauravsz/The-Hub/RouteIQ
source /Users/sauravsz/The-Hub/.venv/bin/activate
./scripts/switch_provider.sh google
streamlit run app.py
# take screenshots and save into assets/, then Ctrl+C

git add README.md assets/baseline-dashboard.png assets/disruption-dashboard.png assets/cost-surge-dashboard.png
git commit -m "Add final dashboard screenshots to README"
git push origin main

git tag -a v1.0.0 -m "RouteIQ v1.0.0: phases 1-7 complete"
git push origin v1.0.0
```
