XAI report layout
=================
sage/
  macro_sage_importance.csv
  classwise_sage_importance.csv
shap/
  global_shap_importance.csv
  classwise_shap_importance.csv
cpf/
  global_cpf_importance.csv
  classwise_cpf_importance.csv
interaction/
  classwise_shap_interactions.csv
lime/
  lime_local_explanations.csv
  classwise_lime_importance.csv
selection/
  classwise_feature_selection_scores.csv
  classwise_selected_features.csv
  final_selected_features.csv

Selection rule
--------------
1. Select the class-balanced Macro-SAGE top-k as the core feature set.
2. Add class protection from the intersection of class-wise SHAP top-k and
   positive class-wise CPFI top-k, with consensus fallback where necessary.
3. Fill to min_final_features in Macro-SAGE rank order.
4. Protect missing partners from valid class-wise SHAP interaction pairs up to
   max_final_features. Unsupported multiclass interaction output is skipped.
5. LIME is optional and is not part of reduction-1 automatic selection.

SAGE note
---------
All training rows participate in every repeat. Missing features use a random
donor permutation of the same training set. Per-class one-vs-rest binary
cross-entropy contributions are balanced 50/50 between positive and negative
rows, then averaged equally across classes to obtain Macro-SAGE.

CPFI note
---------
The CPFI implementation is an approximation: each feature is shuffled inside
quantile bins formed by its strongest correlated features. It does not sample
from the exact full conditional distribution p(X_j | X_-j).
