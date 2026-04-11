// Vercel Serverless Function: /api/statistics
// Statistics endpoint for dashboard

module.exports = async (req, res) => {
  // Allow GET and HEAD
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const MHRAS_API_URL = process.env.MHRAS_API_URL;

    // If backend URL is configured, proxy to FastAPI
    if (MHRAS_API_URL) {
      const getFetch = () => {
        if (typeof fetch !== 'undefined') return fetch;
        if (typeof global !== 'undefined' && typeof global.fetch !== 'undefined') return global.fetch;
        try {
          return require('node-fetch');
        } catch (_e) {
          return null;
        }
      };

      const fetchFn = getFetch();
      if (!fetchFn) {
        return res.status(500).json({ error: 'Fetch unavailable' });
      }

      try {
        const backendResponse = await fetchFn(`${MHRAS_API_URL}/statistics`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            ...(req.headers.authorization && { Authorization: req.headers.authorization })
          }
        });

        if (backendResponse.ok) {
          const data = await backendResponse.json();
          return res.status(200).json(data);
        }
      } catch (e) {
        // Fall through to demo mode response
      }
    }

    // Demo mode: return simulated statistics
    return res.status(200).json({
      models: {
        active_count: 2,
        total_count: 2,
        model_types: ['logistic_regression', 'lightgbm'],
        last_trained: new Date(Date.now() - 86400000).toISOString()
      },
      review_queue: {
        pending_count: 3,
        total_count: 47,
        overdue_count: 0,
        average_review_time_hours: 2.5,
        oldest_pending_hours: 4.2
      },
      screenings: {
        total_today: 127,
        total_week: 842,
        risk_distribution: {
          low: 62,
          moderate: 24,
          high: 10,
          critical: 4
        }
      },
      system: {
        mode: 'demo',
        version: '1.0.0',
        uptime_seconds: Math.floor(process.uptime?.() || 0)
      }
    });

  } catch (error) {
    console.error('Statistics error:', error);
    return res.status(500).json({
      error: 'Internal server error',
      detail: error.message
    });
  }
};