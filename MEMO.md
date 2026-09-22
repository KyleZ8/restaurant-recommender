# Launch Recommendation

## Recommendation

Launch **matrix factorization for warm users**, backed by a **popularity fallback** for new users and new restaurants. The offline evidence is mixed: matrix factorization has the best overall RMSE (**1.4020**) and highest catalog coverage (**1.44%**), but every model has **0.0% hit rate@10** on the executed sparse sample. Treat this as a cautious experiment, not a full-confidence launch.

## Hybrid Rule

- If the user has prior rating history and the restaurant has training history, serve matrix-factorization recommendations.
- If the user is new, the restaurant is new, or the model cannot score reliably, fall back to popularity within the product's normal location/category filters.
- Monitor popularity concentration: matrix factorization sends **19.6%** of slots to the top 10% most-reviewed restaurants.

## Risks

Cold start is the main risk. For new restaurants, popularity RMSE is **1.2398**, while the personalized models are around **1.4503**. For warm users, matrix factorization is better (**1.2983** RMSE vs. **1.3928** for popularity), so the product should use personalization only where history exists.

## A/B Test Plan

Run a user-level A/B test: control is the current popularity fallback, treatment is matrix factorization with the fallback rule above. Primary metric: recommendation click-through or save rate. Guardrails: restaurant diversity, share of slots going to top 10% restaurants, low-rated recommendation rate, latency, and no drop in conversion to restaurant detail page. For a rough sizing target, plan for about **10k users per arm** before reading results; refine with baseline CTR and minimum detectable lift once production traffic is known.
