# Launch Recommendation

## Recommendation

Launch the **popularity baseline first**, then test matrix factorization as a warm-user reranker. On a deterministic 5,000-user sample that preserves all selected users' reviews, popularity has the best hit rate@10 (**10.0%**) while matrix factorization has the best RMSE (**1.2142**). Treat the launch as guarded because the hit-rate winner also sends **100.0%** of recommendation slots to the top 10% most-reviewed restaurants.

## Hybrid Rule

- Use popularity as the default top-10 candidate generator inside the product's normal location/category filters.
- For warm users, A/B test matrix factorization as a reranker or blended candidate source because it is the better rating predictor.
- Monitor popularity concentration: popularity sends **100.0%** of slots to the top 10% most-reviewed restaurants, while matrix factorization sends **72.3%**.

## Risks

Popularity concentration is the main risk. The corrected hit-rate convention is user-level, not row-level: popularity hits **10.0%** of evaluated users, while matrix factorization hits **3.0%**. For warm users, matrix factorization is still better on RMSE (**1.1885** vs. **1.3103** for popularity), so the product should test personalization where history exists and cap repeated top-popularity exposure.

## A/B Test Plan

Run a user-level A/B test: control is popularity, treatment is a matrix-factorization reranker or blend for warm users. Primary metric: recommendation click-through or save rate. Guardrails: restaurant diversity, share of slots going to top 10% restaurants, low-rated recommendation rate, latency, and no drop in conversion to restaurant detail page. For a rough sizing target, plan for about **10k users per arm** before reading results; refine with baseline CTR and minimum detectable lift once production traffic is known.
