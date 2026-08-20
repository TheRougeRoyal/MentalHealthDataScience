# 🚀 MindMetrics AI - Quick Start Guide

## Get Running in 30 Seconds

### Option 1: Direct File Open (Simplest)
```bash
# Just double-click this file:
index.html

# That's it! The app opens in your browser.
```

### Option 2: Local Web Server (Recommended)
```bash
# Python 3:
python -m http.server 8080

# Python 2:
python -m SimpleHTTPServer 8080

# Node.js:
npx serve .

# Then open: http://localhost:8080
```

---

## 🎯 First Time User Path

### 1️⃣ Try Individual Assessment (2 minutes)

1. **Open the app** → You'll see the dashboard
2. **Scroll to "Screening Tool"** section
3. **Click "Autofill Test Sample"** → Form fills automatically
4. **Check the consent box**
5. **Click "Run Statistical Risk Model"** → Results appear in seconds
6. **Explore results:**
   - Risk score gauge (visual)
   - Contributing factors (text)
   - What-if simulator (interactive)

### 2️⃣ Try Batch Analytics (1 minute)

1. **Navigate to "Batch Analytics"** section
2. **Click "Load Sample Batch"** → JSON data appears
3. **Click "Execute Batch Analysis"** → Instant results
4. **See the magic:**
   - Color distribution chart
   - Individual result cards
   - Export buttons appear
5. **Click "Export Results as CSV"** → Download file

### 3️⃣ Explore Statistical Dashboard (30 seconds)

1. **Look at the top cards** → See auto-generated stats
2. **Scroll to "Statistical Analysis"** section → See 150 mock records
3. **Click "Refresh Analysis"** → Watch new data generate
4. **Observe changes** → Different averages, distributions

---

## 🎨 Cool Features to Try

### Interactive What-If Simulator
**After completing an assessment:**
1. Scroll to "What-If Feature Simulator"
2. Move the PHQ-9 slider left (decrease depression)
3. Watch risk score drop in real-time
4. See delta indicator: "▼ 15.3 pts lower"

### Risk Trajectory
**Multiple assessments with same ID:**
1. Use ID: `patient_test_001`
2. Submit assessment with PHQ-9: 15
3. Submit again with PHQ-9: 10
4. See trend chart appear showing improvement

### Export Your Data
**After batch analysis:**
1. Click "📊 Export Results as CSV"
2. Open in Excel/Google Sheets
3. Analyze risk distributions
4. Share with team

### Theme Toggle
1. Click theme button (top right / mobile menu)
2. Switch between dark and light modes
3. Refresh page → preference persists

---

## 📱 Mobile Experience

### On Your Phone:
1. Open `index.html` in mobile browser
2. Tap hamburger menu (☰) in top-right
3. Drawer slides out with navigation
4. Tap "Batch Analytics" → Menu auto-closes
5. Everything works perfectly!

---

## 🧪 Testing Scenarios

### Quick Test: All Risk Levels

**Low Risk Patient:**
```
PHQ-9: 3
GAD-7: 2
Sleep: 8 hours
Heart Rate: 65 bpm
→ Score: ~15-20 (Green badge)
```

**Critical Risk Patient:**
```
PHQ-9: 24
GAD-7: 19
Sleep: 3.5 hours
Heart Rate: 95 bpm
→ Score: ~80-90 (Dark red badge)
→ Alert flag appears
→ Review required flag appears
```

---

## 🎯 5-Minute Complete Tour

### Minute 1: Dashboard Overview
- Look at 4 stat cards at top
- Notice "UI Mode" status (means using mock data)
- See statistical analysis section

### Minute 2: Individual Assessment
- Click "Screening Tool" in sidebar
- Click "Autofill Test Sample"
- Submit assessment
- Review all result sections

### Minute 3: Batch Processing
- Navigate to "Batch Analytics"
- Load sample batch
- Execute analysis
- Explore distribution chart

### Minute 4: Simulator & Visualization
- Return to individual results
- Try what-if simulator
- Observe risk gauge animation
- Check feature attribution chart

### Minute 5: Export & Review
- Export batch results as CSV
- Open review queue
- Select a demo case
- Explore review workflow

---

## ❓ Troubleshooting

### "Nothing happens when I click buttons"
**Solution**: Check browser console (F12) for errors. Ensure JavaScript is enabled.

### "Sample data doesn't load"
**Solution**: Refresh page. Check if `app.js` is loaded (view source, Ctrl+U).

### "Export doesn't download"
**Solution**: Check browser's download settings. Allow downloads from local files.

### "Looks broken on mobile"
**Solution**: Try different browser (Chrome/Safari). Ensure screen width < 768px for mobile view.

### "Statistical analysis shows 0 records"
**Solution**: Click "Refresh Analysis" button. Should generate 150 new records.

---

## 🎓 Learning Path

### For Beginners:
1. ✅ Try individual assessment with sample data
2. ✅ Understand risk levels (Low/Moderate/High/Critical)
3. ✅ Explore what-if simulator
4. ✅ Read contributing factors

### For Intermediate:
1. ✅ Create custom batch JSON data
2. ✅ Export and analyze CSV results
3. ✅ Compare different patient scenarios
4. ✅ Understand risk model weights

### For Advanced:
1. ✅ Open browser DevTools and inspect `clientScore()` function
2. ✅ Review mock data generation algorithm
3. ✅ Analyze force plot attribution
4. ✅ Modify risk thresholds in code

---

## 📚 Where to Go Next

### Want More Details?
→ **DEMO.md** - Complete feature walkthrough with screenshots

### Want Technical Info?
→ **README.md** - Full architecture and API documentation

### Want Feature List?
→ **FEATURES.md** - Every feature explained

### Want to Test?
→ **validate.html** - Automated feature testing page

### Want Change History?
→ **CHANGES.md** - What was built and why

---

## 💡 Pro Tips

### Tip 1: Keyboard Shortcuts
- `Tab` - Navigate through form fields
- `Enter` - Submit forms
- `Esc` - Close mobile menu

### Tip 2: Use Same Patient ID
- Reuse IDs to build trend history
- Track risk over time
- See trajectory visualization

### Tip 3: Custom Batch Data
- Start with sample batch
- Modify values for your scenarios
- Test edge cases (very high/low scores)

### Tip 4: Export Everything
- Export batch results for analysis
- Share CSV with team
- Archive JSON for records

### Tip 5: Mobile Demo
- Great for tablet demos
- Touch-friendly interface
- Impressive in meetings

---

## 🎉 Success Checklist

After 5 minutes, you should have:
- [ ] Completed 1 individual assessment
- [ ] Seen your risk score and level
- [ ] Tried the what-if simulator
- [ ] Run 1 batch analysis
- [ ] Exported results to CSV
- [ ] Viewed statistical dashboard
- [ ] Toggled theme at least once
- [ ] Tried on mobile (optional)

**All checked?** You're ready to use MindMetrics AI! 🎊

---

## 🚀 Next Actions

### For Demos:
1. Practice the 5-minute tour above
2. Prepare custom patient scenarios
3. Have `validate.html` open as backup
4. Test on presentation laptop/tablet

### For Development:
1. Review `app.js` source code
2. Understand `clientScore()` algorithm
3. Explore React/Vue integration options
4. Plan backend API connection

### For Production:
1. Set up Firebase project
2. Configure backend API
3. Deploy to Vercel/Netlify
4. Enable authentication

---

## 📞 Need Help?

### Common Questions:
**Q: Does this need internet?**  
A: No! Works 100% offline.

**Q: Is my data saved?**  
A: No, UI-only mode doesn't persist data. Results exist only in browser session.

**Q: Can I use real patient data?**  
A: Yes, but review legal disclaimers first. Consider HIPAA compliance for production.

**Q: How accurate is the risk model?**  
A: Based on clinical guidelines (PHQ-9, GAD-7). For research/education, not diagnosis.

**Q: Can I customize thresholds?**  
A: Yes! Edit `clientScore()` function in `app.js`.

---

**🎯 Start now: Just open `index.html` and click around!**

**Everything works out of the box. No setup. No configuration. Just explore.** ✨
