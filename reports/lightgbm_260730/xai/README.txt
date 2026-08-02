XAI report layout
=================
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
1. Strictly select the intersection of class-wise SHAP top-k and positive
   class-wise CPFI top-k.
2. If a class has too few intersecting features, fill by the combined
   normalized SHAP/CPFI consensus score.
3. Protect a missing partner when valid output-specific class-wise SHAP
   interaction tensors are available. Unsupported multiclass interaction
   outputs are skipped rather than duplicated across classes.
4. Optionally add top local LIME features when include_lime_in_selection=True.

CPFI note
---------
The CPFI implementation is an approximation: each feature is shuffled inside
quantile bins formed by its strongest correlated features. It does not sample
from the exact full conditional distribution p(X_j | X_-j).
