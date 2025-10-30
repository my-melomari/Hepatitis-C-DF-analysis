import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.manifold import TSNE
from IPython.display import display, HTML

display(HTML('''
<div style="
    border: 2px solid #f8c8dc;
    background-color: #ffe6ef;
    padding: 14px 18px;
    border-radius: 12px;
    font-weight: bold;
    color: #3D3D3D;
    font-size: 15px;
    font-family: 'Segoe UI', sans-serif;
">
I admit that I do not understand the full extent of Hepatitis C virus pathology nor its implications,
so the observations are made based on my own research.
Please take it with a grain of salt.
</div>
'''))

# Prepare data
df = pd.read_csv("hcvdat0.csv").drop(columns=["Unnamed: 0"])

num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

num_imputer = SimpleImputer(strategy='median')
if len(num_cols) > 0:
	df[num_cols] = num_imputer.fit_transform(df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
if len(cat_cols) > 0:
	df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

df["Sex"] = LabelEncoder().fit_transform(df["Sex"]) 
df["Category"] = df["Category"].str.replace("0=", "").str.strip()

# Scale numeric features
features = df.drop(columns=["Category"])
X = StandardScaler().fit_transform(features)
y = df["Category"]

n_categories = len(df['Category'].unique())
pastel_palette = sns.color_palette("pastel", n_colors=n_categories)
pink_palette = sns.color_palette("RdPu", n_colors=n_categories)

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(pd.DataFrame(X, columns=features.columns).corr(), 
            annot=True, fmt=".2f", 
            cmap=sns.light_palette("pink", as_cmap=True))
plt.title("Correlation heatmap")
plt.tight_layout()
plt.show()

bio_summary = pd.DataFrame({
    "Relationship": ["ALB–PROT", "CHE–CHOL", "GGT–AST / ALP", "BIL–CHE"],
    "Correlation": ["+0.55", "+0.42", "+0.44–0.49", "–0.33"],
    " ": [
        "Albumin contributes directly to total protein",
        "Both reflect hepatic synthetic capacity",
        "Indicators of hepatocellular/cholestatic injury",
        "Inverse pattern between excretory and synthetic function"
    ]
})

print("Biological data interpretation\n")
print(bio_summary.to_string(index=False))

# PCA for linear patterns
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette=pastel_palette)
plt.title("PCA projection by diagnostic category")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.tight_layout()
plt.show()

pca_summary = pd.DataFrame({
    "Component": ["PC1", "PC2", "PC1 + PC2"],
    "Explained variance (%)": [20.5, 15.8, 36.3],
    "Interpretation": [
        "Greatest variance in biomarkers, primary metabolic differences.",
        "Orthogonal to PC1, complementary variation patterns.",
        "Total variance — moderate dimensional complexity."
    ]
})

cluster_summary = pd.DataFrame({
    " ": ["Blood donors", "Hepatitis, fibrosis, cirrhosis", 
              "cirrhosis & fibrosis", "Suspect blood donors"],
    " ": [
        "Tight, compact cluster suggesting consistent, normal biomarker profiles.",
        "More dispersed distribution, greater variability among patients.",
        "Spread toward negative PC1/PC2 region, distinct metabolic and enzymatic shifts.",
        "Located between healthy and disease clusters, representing early cases."
    ]
})

print("\nPCA interpretation\n")
print(pca_summary.to_string(index=False))

print("\nCluster interpretation")
print(cluster_summary.to_string(index=False))

X_df = pd.DataFrame(X, columns=features.columns)
X_df['Category'] = y
category_means = X_df.groupby('Category').mean()

categories = category_means.columns
num_vars = len(categories)
angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))


for idx, category in enumerate(category_means.index):
    values = category_means.loc[category].values
    values = np.concatenate((values, [values[0]]))
    color = pastel_palette[idx % len(pastel_palette)]
    ax.plot(angles, values, 'o-', linewidth=2, label=category, color=color)
    ax.fill(angles, values, alpha=0.25, color=color)


ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(-3, 3)

plt.title("Biomarkers by category", pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

data = {
    " ": ["Blood donors", "Suspect donors", "Hepatitis", "Fibrosis", "Cirrhosis"],
    " ": [
        "Stable and balanced across all biomarkers.",
        "Slight rise in ALT and GGT compared to healthy donors.",
        "Elevated ALT and AST indicating liver inflammation.",
        "Broader increases in AST and GGT with mild CHE and ALB drop.",
        "High AST and BIL with low ALB and CHE, showing severe liver damage."
    ],
}

summary = pd.DataFrame(data)

print("Summary")
print(summary.to_string(index=False))

# Random forest for feature importance
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=features.columns).sort_values(ascending=False)

plt.figure(figsize=(8,6))
ax = sns.barplot(x=importances.values, y=importances.index, hue=importances.index, 
                 palette=sns.light_palette("pink", n_colors=len(importances), reverse=True))
if ax.get_legend() is not None:
    ax.get_legend().remove()
plt.title("Feature importance (Random Forest)")
plt.xlabel("Importance score")
plt.ylabel("Biochemical marker")
plt.tight_layout()
plt.show()

feature_importance_summary = pd.DataFrame({
    "Rank": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    ],
    "Marker": [
        "AST", "ALT", "CHE", "ALP", "GGT", "ALB", "BIL", "PROT", "Age", "CHOL/CREA", "Sex"
    ],
    " ": [
        "highly elevated in hepatocellular injury and cirrhosis.",
        "Hepatocellular enzyme rising early in hepatitis and inflammation.",
        "Declines in chronic liver damage.",
        "Indicator of biliary obstruction or cholestatic injury.",
        "Supports ALP in identifying cholestatic versus hepatocellular damage.",
        "Advanced liver disease due to reduced protein synthesis.",
        "Excretory dysfunction and jaundice progression in cirrhosis.",
        "Total protein level, moderately sensitive to liver failure.",
        "Age-related biomarker variability and susceptibility effects.",
        "Broader metabolic and renal interactions with hepatic function.",
        "Minimal contribution."
    ]
})

print(feature_importance_summary.to_string(index=False))

# Pairwise biomarker relationship
sns.pairplot(df, vars=["ALB","ALP","ALT","AST","BIL","CHOL"], 
            hue="Category", diag_kind="kde", 
            palette=pastel_palette,
            plot_kws={'alpha': 0.6},
            diag_kws={'fill': False, 'common_norm': False})
plt.suptitle("Pairwise biomarker relationship", y=1.02)
plt.show()

biomarker_relationships = pd.DataFrame({
    " ": [
        "ALT–AST",
        "ALP–GGT",
        "ALB–BIL",
        "CHOL–CHE",
        "ALT–CHOL",
        "BIL–AST/ALT"
    ],
    " ": [
        "Strongly correlated elevations in diseased groups, especially fibrosis and cirrhosis.",
        "Moderate co-elevation across cholestatic and advanced liver disease clusters.",
        "Low albumin coincides with high bilirubin in cirrhosis cases.",
        "Positive correlation in healthy donors but disrupted in fibrosis and cirrhosis.",
        "Mild negative correlation in disease clusters, reflecting altered metabolism.",
        "Weak to moderate positive correlation in hepatitis and cirrhosis."
    ],
})

print("\nPairwise biomarker relationship\n")
print(biomarker_relationships.to_string(index=False))