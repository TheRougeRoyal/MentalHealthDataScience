// Vercel Serverless Function: /api/screen
// Proxies to FastAPI backend when MHRAS_API_URL is configured

const getFetch = () => {
  if (typeof fetch !== 'undefined') return fetch;
  if (typeof global !== 'undefined' && typeof global.fetch !== 'undefined') return global.fetch;
  try {
    return require('node-fetch');
  } catch (_e) {
    return null;
  }
};

module.exports = async (req, res) => {
  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Parse body safely - Vercel handles JSON parsing, but handle edge cases
    let requestBody;
    try {
      requestBody = typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
    } catch (parseError) {
      return res.status(400).json({ error: 'Invalid JSON in request body' });
    }

    if (!requestBody || typeof requestBody !== 'object') {
      return res.status(400).json({ error: 'Request body must be JSON' });
    }

    // Validate required fields
    if (!requestBody.anonymized_id) {
      return res.status(400).json({ error: 'anonymized_id is required' });
    }

    if (!requestBody.consent_verified) {
      return res.status(403).json({ error: 'Consent must be verified' });
    }

    // If backend URL is configured, proxy to FastAPI
    const MHRAS_API_URL = process.env.MHRAS_API_URL;
    if (MHRAS_API_URL) {
      const fetchFn = getFetch();
      if (!fetchFn) {
        return res.status(500).json({ error: 'Fetch API unavailable. Ensure Node 18+ runtime or add node-fetch dependency.' });
      }

      const backendResponse = await fetchFn(`${MHRAS_API_URL}/screen`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(req.headers.authorization && { Authorization: req.headers.authorization })
        },
        body: JSON.stringify(requestBody)
      });

      if (!backendResponse.ok) {
        const error = await backendResponse.json();
        return res.status(backendResponse.status).json(error);
      }

      const data = await backendResponse.json();
      return res.status(200).json(data);
    }

    // Demo mode: return simulated response when no backend is configured
    const surveyData = requestBody.survey_data || {};
    const phq9Score = surveyData.phq9_score || 0;
    const gad7Score = surveyData.gad7_score || 0;

    // Calculate simulated risk score
    let riskScore = 30;
    if (phq9Score > 15) riskScore += 25;
    else if (phq9Score > 10) riskScore += 15;
    else if (phq9Score > 5) riskScore += 5;

    if (gad7Score > 15) riskScore += 20;
    else if (gad7Score > 10) riskScore += 10;
    else if (gad7Score > 5) riskScore += 5;

    riskScore = Math.min(100, Math.max(0, riskScore));

    let riskLevel = 'LOW';
    if (riskScore > 75) riskLevel = 'CRITICAL';
    else if (riskScore > 60) riskLevel = 'HIGH';
    else if (riskScore > 40) riskLevel = 'MODERATE';

    const contributingFactors = [];
    if (phq9Score > 10) contributingFactors.push('Elevated PHQ-9 depression score');
    if (gad7Score > 10) contributingFactors.push('Elevated GAD-7 anxiety score');
    if (requestBody.wearable_data?.sleep_hours < 5) contributingFactors.push('Poor sleep duration');
    if (!contributingFactors.length) contributingFactors.push('No high-risk indicators detected');

    const recommendations = [];
    if (riskLevel === 'CRITICAL') {
      recommendations.push({
        resource_type: 'crisis_line',
        name: '988 Suicide & Crisis Lifeline',
        description: '24/7 crisis support - Call or Text 988',
        contact_info: 'Call or Text: 988',
        urgency: 'immediate',
        eligibility_criteria: {}
      });
    }
    if (riskLevel === 'HIGH' || riskLevel === 'CRITICAL') {
      recommendations.push({
        resource_type: 'therapy',
        name: 'Cognitive Behavioral Therapy (CBT)',
        description: 'Evidence-based therapy for depression and anxiety',
        contact_info: 'Contact local mental health provider',
        urgency: 'soon',
        eligibility_criteria: {}
      });
    }
    if (riskLevel === 'MODERATE' || riskLevel === 'LOW') {
      recommendations.push({
        resource_type: 'wellness',
        name: 'Sleep Hygiene Education',
        description: 'Improve sleep habits for better mental health',
        contact_info: 'Contact healthcare provider',
        urgency: 'routine',
        eligibility_criteria: {}
      });
    }

    return res.status(200).json({
      risk_score: {
        anonymized_id: requestBody.anonymized_id,
        score: riskScore,
        risk_level: riskLevel,
        confidence: 0.75,
        contributing_factors: contributingFactors,
        timestamp: new Date().toISOString()
      },
      recommendations: recommendations,
      explanations: {
        top_features: [
          ['phq9_score', phq9Score * 0.15],
          ['gad7_score', gad7Score * 0.12]
        ],
        counterfactual: riskLevel !== 'LOW'
          ? `If PHQ-9 were ${Math.max(0, phq9Score - 5)} and GAD-7 were ${Math.max(0, gad7Score - 3)}, risk would decrease.`
          : 'Current indicators suggest stable mental health.',
        rule_approximation: '',
        clinical_interpretation: `PHQ-9 of ${phq9Score} suggests ${phq9Score < 5 ? 'minimal' : phq9Score < 10 ? 'mild' : phq9Score < 15 ? 'moderate' : phq9Score < 20 ? 'moderately severe' : 'severe'} depression. GAD-7 of ${gad7Score} suggests ${gad7Score < 5 ? 'minimal' : gad7Score < 10 ? 'mild' : gad7Score < 15 ? 'moderate' : 'severe'} anxiety.`
      },
      requires_human_review: riskScore > 75,
      alert_triggered: riskScore > 85
    });

  } catch (error) {
    console.error('Screening error:', error);
    return res.status(500).json({ error: 'Internal server error', detail: error.message });
  }
};