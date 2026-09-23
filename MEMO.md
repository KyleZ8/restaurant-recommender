# Launch Recommendation

## Recommendation

Launch **matrix factorization for warm users**, backed by a **popularity fallback** for new users and new restaurants. On a deterministic 5,000-user sample that preserves all selected users' reviews, matrix factorization has the best RMSE (**1.2142**) and the only nonzero hit rate@10 (**3.0%**). Treat this as a guarded experiment because it also concentrates recommendations heavily in already-popular restaurants.

## Hybrid Rule

- If the user has prior rating history and the restaurant has training history, serve matrix-factorization recommendations.
- If the user is new, the restaurant is new, or the model cannot score reliably, fall back to popularity within the product's normal location/category filters.
- Monitor popularity concentration: matrix factorization sends **72.3%** of slots to the top 10% most-reviewed restaurants.

## Risks

Cold start and popularity concentration are the main risks. For new users, every model collapses to the same no-history estimate (**1.2428** RMSE). For warm users, matrix factorization is better (**1.1885** RMSE vs. **1.3103** for popularity), so the product should use personalization only where history exists and should cap or diversify repeated top-popularity exposure.

## A/B Test Plan

Run a user-level A/B test: control is the current popularity fallback, treatment is matrix factorization with the fallback rule above. Primary metric: recommendation click-through or save rate. Guardrails: restaurant diversity, share of slots going to top 10% restaurants, low-rated recommendation rate, latency, and no drop in conversion to restaurant detail page. For a rough sizing target, plan for about **10k users per arm** before reading results; refine with baseline CTR and minimum detectable lift once production traffic is known.
