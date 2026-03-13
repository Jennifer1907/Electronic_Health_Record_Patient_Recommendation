
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.special import expm1
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import warnings
import io

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="HAC Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)


class OutcomeScaler:
    """
    Scale treatment effects to comparable monetary units using WTP values.
    """
    
    def __init__(self):
        self.wtp_longterm_pain = 25000  
        self.wtp_treatment_pain = 15000 
        self.wtp_function = 25000 
        
    def scale_treatment_effects(self, te_row: pd.Series) -> Dict[str, float]:
        te1 = te_row.get('TE1', 0)  
        te2 = te_row.get('TE2', 0)  
        te3 = te_row.get('TE3', 0)  
        te4 = te_row.get('TE4 (Weeks)', te_row.get('TE4', 0))  
        te5 = te_row.get('TE5 ($)', te_row.get('TE5', 0))  
        
        pain_improvement_value = te1 * self.wtp_longterm_pain
        function_improvement_value = te2 * self.wtp_function
        
        treatment_pain_cost = -te4 * (self.wtp_treatment_pain / 52)  
        
        additional_quality_value = te3 * self.wtp_longterm_pain * 0.5 
        
        direct_cost = -te5
        
        return {
            'pain_improvement_value': pain_improvement_value,
            'function_improvement_value': function_improvement_value,
            'treatment_pain_cost': treatment_pain_cost,
            'additional_quality_value': additional_quality_value,
            'direct_cost': direct_cost
        }


class TreatmentRecommender:
    """
    WTP-based treatment recommender with preference integration
    """
    
    def __init__(self, df_treatment_effects: pd.DataFrame, df_patient_prefs: pd.DataFrame):
        self.te_df = df_treatment_effects.copy()
        self.pref_df = df_patient_prefs.copy()
        
        if 'PAT_ID' not in self.te_df.columns:
            self.te_df['PAT_ID'] = range(1, len(self.te_df) + 1)
        if 'PAT_ID' not in self.pref_df.columns:
            self.pref_df['PAT_ID'] = range(1, len(self.pref_df) + 1)
        
        preference_cols = ['WP1', 'WP2', 'WP3', 'WP4', 'WP5']
        self.pref_df['WP_sum'] = self.pref_df[preference_cols].sum(axis=1)
        
        for col in preference_cols:
            self.pref_df[f'{col}_normalized'] = (self.pref_df[col] / self.pref_df['WP_sum']) * 100
        
        self.scaler = OutcomeScaler()
        self.te_cols = [c for c in self.te_df.columns if c.startswith('TE')]
        
        self.linkage_matrix = None
        self.cluster_labels = None
        self.cluster_utilities = {}
        
    def build_clusters(self, n_clusters: int = 5, linkage_method: str = 'ward', max_samples: int = 5000):
        """Build HAC clusters with efficient sampling for large datasets"""
        te_features = self.te_df[self.te_cols].values
        n_patients = len(te_features)
        
        te_mean = te_features.mean(axis=0)
        te_std = te_features.std(axis=0) + 1e-8
        te_std_features = (te_features - te_mean) / te_std
        
        if n_patients > max_samples:
            sample_indices = np.random.choice(n_patients, max_samples, replace=False)
            sample_features = te_std_features[sample_indices]
            self.linkage_matrix = linkage(sample_features, method=linkage_method)
            
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.cluster_labels = kmeans.fit_predict(te_std_features)
            self.te_df['cluster'] = self.cluster_labels + 1  
            
        else:
            self.linkage_matrix = linkage(te_std_features, method=linkage_method)
            self.cluster_labels = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
            self.te_df['cluster'] = self.cluster_labels
        
        return self
    
    def compute_cluster_utilities(self, sample_size: int = 10000):
        merged = self.te_df.merge(self.pref_df, on='PAT_ID', how='inner')
        
        if len(merged) > sample_size:
            merged = merged.sample(n=sample_size, random_state=42)
        
        for cluster_id in sorted(merged['cluster'].unique()):
            cluster_data = merged[merged['cluster'] == cluster_id]
            
            utilities = []
            for _, row in cluster_data.iterrows():
                rec = self.recommend(row, row)
                utilities.append(rec['utility_difference'])
            
            self.cluster_utilities[cluster_id] = {
                'mean': np.mean(utilities) if utilities else 0.0,
                'std': np.std(utilities) if utilities else 0.0,
                'count': len(utilities)
            }
        
        return self
    
    def recommend(self, te_row: pd.Series, pref_row: pd.Series) -> Dict:
        scaled_outcomes = self.scaler.scale_treatment_effects(te_row)
        
        pref_weights = [pref_row.get(f'WP{i}_normalized', 
                                     pref_row.get(f'WP{i}', 20)) for i in range(1, 6)]
        
        components = {
            'pain_improvement_value': {
                'value': scaled_outcomes['pain_improvement_value'],
                'weight': pref_weights[0] / 100.0,
                'contribution': scaled_outcomes['pain_improvement_value'] * (pref_weights[0] / 100.0)
            },
            'function_improvement_value': {
                'value': scaled_outcomes['function_improvement_value'],
                'weight': pref_weights[1] / 100.0,
                'contribution': scaled_outcomes['function_improvement_value'] * (pref_weights[1] / 100.0)
            },
            'treatment_pain_cost': {
                'value': scaled_outcomes['treatment_pain_cost'],
                'weight': pref_weights[2] / 100.0,
                'contribution': scaled_outcomes['treatment_pain_cost'] * (pref_weights[2] / 100.0)
            },
            'additional_quality_value': {
                'value': scaled_outcomes['additional_quality_value'],
                'weight': pref_weights[3] / 100.0,
                'contribution': scaled_outcomes['additional_quality_value'] * (pref_weights[3] / 100.0)
            },
            'direct_cost': {
                'value': scaled_outcomes['direct_cost'],
                'weight': pref_weights[4] / 100.0,
                'contribution': scaled_outcomes['direct_cost'] * (pref_weights[4] / 100.0)
            }
        }
        
        total_utility = sum(comp['contribution'] for comp in components.values())
        if total_utility > 0:
            recommended_treatment = 'Surgery'
        else:
            recommended_treatment = 'Conservative Care'
        
        abs_utility = abs(total_utility)
        if abs_utility > 5000:
            confidence = 'High'
        elif abs_utility > 2000:
            confidence = 'Medium'
        else:
            confidence = 'Low'

        trade_offs = []
        positive_components = [k for k, v in components.items() if v['contribution'] > 0]
        negative_components = [k for k, v in components.items() if v['contribution'] < 0]
        
        if len(positive_components) > 0 and len(negative_components) > 0:
            trade_offs.append(f"Surgery favored by: {', '.join(positive_components)}")
            trade_offs.append(f"Conservative care favored by: {', '.join(negative_components)}")

        qualifications = []
        if confidence == 'Low':
            qualifications.append("Borderline case - patient preferences are critical")
        if abs(components['direct_cost']['contribution']) > 3000:
            qualifications.append("Significant cost difference - discuss financial implications")
        if abs(components['treatment_pain_cost']['contribution']) > 2000:
            qualifications.append("Treatment pain duration is a major factor")
        
        cluster_id = te_row.get('cluster', None) if 'cluster' in te_row else None
        
        return {
            'patient_id': te_row.get('PAT_ID', None),
            'recommended_treatment': recommended_treatment,
            'utility_difference': total_utility,
            'confidence': confidence,
            'utility_details': {
                'components': components,
                'scaled_outcomes': scaled_outcomes
            },
            'cluster': cluster_id,
            'trade_offs': trade_offs,
            'qualifications': qualifications
        }
    
    def recommend_for_patient(self, patient_id: int) -> Dict:
        """Generate recommendation for a specific patient"""
        patient_te = self.te_df[self.te_df['PAT_ID'] == patient_id]
        patient_pref = self.pref_df[self.pref_df['PAT_ID'] == patient_id]
        
        if patient_te.empty or patient_pref.empty:
            return {'error': 'Patient not found'}
        
        te_row = patient_te.iloc[0]
        pref_row = patient_pref.iloc[0]
        
        return self.recommend(te_row, pref_row)


class SensitivityAnalyzer:
    def __init__(self, recommender: TreatmentRecommender):
        self.recommender = recommender
    
    def analyze(self, patient_id: int, variation_pct: float = 0.20) -> Dict:
        """
        Test recommendation stability by varying each preference ±variation_pct
        """
        base_rec = self.recommender.recommend_for_patient(patient_id)
        if 'error' in base_rec:
            return base_rec
        
        base_recommendation = base_rec['recommended_treatment']
        base_utility = base_rec['utility_difference']
        
        variations = {}
        flippable_count = 0

        patient_pref = self.recommender.pref_df[
            self.recommender.pref_df['PAT_ID'] == patient_id
        ].copy()
        
        patient_te = self.recommender.te_df[
            self.recommender.te_df['PAT_ID'] == patient_id
        ].iloc[0]
        
        for i in range(1, 6):
            wp_col = f'WP{i}'
            wp_col_norm = f'WP{i}_normalized'
            original_val = patient_pref[wp_col].values[0]
            
            for direction in ['increase', 'decrease']:
                key = f'WP{i}_{direction}'
                
                if direction == 'increase':
                    new_val = original_val * (1 + variation_pct)
                else:
                    new_val = original_val * (1 - variation_pct)
                
                temp_pref = patient_pref.copy()
                temp_pref[wp_col] = new_val
                
                wp_sum = temp_pref[['WP1', 'WP2', 'WP3', 'WP4', 'WP5']].sum(axis=1).values[0]
                for j in range(1, 6):
                    temp_pref[f'WP{j}_normalized'] = (temp_pref[f'WP{j}'] / wp_sum) * 100

                new_rec = self.recommender.recommend(patient_te, temp_pref.iloc[0])
                new_recommendation = new_rec['recommended_treatment']
                
                changed = (new_recommendation != base_recommendation)
                if changed and direction == 'increase':
                    flippable_count += 1
                
                variations[key] = {
                    'changed': changed,
                    'new_recommendation': new_recommendation,
                    'original_value': original_val,
                    'modified_value': new_val
                }
        
        recommendation_stable = (flippable_count == 0)
        
        if flippable_count == 0:
            if abs(base_utility) > 5000:
                sensitivity_level = "Very Low (Robust)"
            else:
                sensitivity_level = "Low (Stable)"
        elif flippable_count == 1:
            sensitivity_level = "Moderate"
        elif flippable_count == 2:
            sensitivity_level = "Moderate-High"
        elif flippable_count == 3:
            sensitivity_level = "High"
        elif flippable_count == 4:
            sensitivity_level = "High (Fragile)"
        else:
            sensitivity_level = "Very High (Critical)"
        
        return {
            'base_recommendation': base_recommendation,
            'base_utility': base_utility,
            'recommendation_stable': recommendation_stable,
            'num_flippable': flippable_count,
            'sensitivity_level': sensitivity_level,
            'variations': variations
        }


def plot_dendrogram(linkage_matrix, title="Hierarchical Clustering Dendrogram"):
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linkage_matrix, ax=ax, leaf_font_size=8, color_threshold=0)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Patient Index', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_treatment_effects_components(components_dict: Dict, title="Outcome Contributions"):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    outcome_names = ['Pain\nImprovement', 'Function\nImprovement', 'Treatment\nPain',
                     'Quality\nValue', 'Direct\nCost']
    outcome_values = [
        components_dict['pain_improvement_value']['contribution'],
        components_dict['function_improvement_value']['contribution'],
        components_dict['treatment_pain_cost']['contribution'],
        components_dict['additional_quality_value']['contribution'],
        components_dict['direct_cost']['contribution']
    ]
    
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in outcome_values]
    bars = ax.barh(range(5), outcome_values, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(range(5))
    ax.set_yticklabels(outcome_names)
    ax.set_xlabel('Contribution to Utility ($)', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.axvline(0, color='black', linewidth=2)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, outcome_values)):
        width = bar.get_width()
        label_x = width + (200 if width > 0 else -200)
        ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'${val:,.0f}', ha='left' if width > 0 else 'right', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_sensitivity_analysis(sensitivity_result: Dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    preferences = ['Pain', 'Function', 'Tx Pain', 'Quality', 'Cost']
    flippable = [0, 0, 0, 0, 0]
    
    for i in range(1, 6):
        key = f'WP{i}_increase'
        if key in sensitivity_result['variations']:
            flippable[i-1] = 1 if sensitivity_result['variations'][key]['changed'] else 0
    
    colors = ['#ff6b6b' if f == 1 else '#95e1d3' for f in flippable]
    axes[0].barh(preferences, flippable, color=colors)
    axes[0].set_xlabel('Can Flip Decision (1=Yes, 0=No)', fontsize=12)
    axes[0].set_title('Preference Sensitivity at ±20%', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 1.2)
    axes[0].grid(axis='x', alpha=0.3)
    
    stability_text = f"Sensitivity Level:\n{sensitivity_result['sensitivity_level']}\n\n"
    stability_text += f"Flippable: {sensitivity_result['num_flippable']}/5\n\n"
    
    if sensitivity_result['recommendation_stable']:
        stability_text += "STABLE\nRecommendation robust\nto preference variations"
        bgcolor = '#d4edda'
    else:
        stability_text += "UNSTABLE\nVerify preferences\ncarefully with patient"
        bgcolor = '#fff3cd'
    
    axes[1].text(0.5, 0.5, stability_text, ha='center', va='center',
                fontsize=14, family='monospace',
                bbox=dict(boxstyle='round', facecolor=bgcolor, alpha=0.8, pad=2))
    axes[1].axis('off')
    axes[1].set_title('Overall Stability', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_comprehensive_dashboard(patient_id: int, 
                                   recommender: TreatmentRecommender,
                                   sensitivity_analyzer: SensitivityAnalyzer):
    rec = recommender.recommend_for_patient(patient_id)
    sens = sensitivity_analyzer.analyze(patient_id)
    
    if 'error' in rec:
        st.error(f"Error: {rec['error']}")
        return None, None
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    ax_main = fig.add_subplot(gs[0, 0])
    ax_main.axis('off')
    
    rec_color = '#2ecc71' if rec['recommended_treatment'] == 'Surgery' else '#3498db'
    main_text = f"RECOMMENDED:\n{rec['recommended_treatment']}\n\nConfidence: {rec['confidence']}"
    
    ax_main.text(0.5, 0.5, main_text, ha='center', va='center',
                fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=rec_color, alpha=0.3, pad=2))
    ax_main.set_title(f'Patient {patient_id} - Recommendation',
                     fontsize=14, fontweight='bold')
    
    ax_pref = fig.add_subplot(gs[0, 1])
    patient_pref = recommender.pref_df[recommender.pref_df['PAT_ID'] == patient_id].iloc[0]
    pref_names = ['Pain\nImprov', 'Function\nImprov', 'Treat\nPain', 'Quality', 'Cost']
    pref_values = [patient_pref[f'WP{i}_normalized'] for i in range(1, 6)]
    
    ax_pref.bar(range(5), pref_values, color='steelblue', alpha=0.7, edgecolor='black')
    ax_pref.set_xticks(range(5))
    ax_pref.set_xticklabels(pref_names, fontsize=9)
    ax_pref.set_ylabel('Preference Weight (%)', fontweight='bold')
    ax_pref.set_title('Patient Preferences', fontweight='bold')
    ax_pref.grid(axis='y', alpha=0.3)
    ax_pref.set_ylim(0, max(pref_values) * 1.2)
    
    ax_util = fig.add_subplot(gs[0, 2])
    utility_val = rec['utility_difference']
    surgery_util = max(0, utility_val)
    conservative_util = max(0, -utility_val)
    
    bars = ax_util.barh(['Conservative', 'Surgery'], 
                        [conservative_util, surgery_util],
                        color=['#3498db', '#2ecc71'], edgecolor='black', linewidth=1.5)
    ax_util.set_xlabel('Utility Value ($)', fontweight='bold')
    ax_util.set_title('Overall Utility Comparison', fontweight='bold')
    ax_util.grid(axis='x', alpha=0.3)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if width > 0:
            ax_util.text(width, bar.get_y() + bar.get_height()/2, 
                        f'${width:,.0f}', ha='left', va='center', fontweight='bold')
    
    ax_outcomes = fig.add_subplot(gs[1, :])
    
    components = rec['utility_details']['components']
    outcome_names = ['Pain\nImprovement', 'Function\nImprovement', 'Treatment\nPain',
                     'Quality\nValue', 'Direct\nCost']
    outcome_values = [
        components['pain_improvement_value']['contribution'],
        components['function_improvement_value']['contribution'],
        components['treatment_pain_cost']['contribution'],
        components['additional_quality_value']['contribution'],
        components['direct_cost']['contribution']
    ]
    
    colors_outcomes = ['#2ecc71' if v > 0 else '#e74c3c' for v in outcome_values]
    bars = ax_outcomes.barh(range(5), outcome_values, color=colors_outcomes, 
                           edgecolor='black', linewidth=1.5)
    ax_outcomes.set_yticks(range(5))
    ax_outcomes.set_yticklabels(outcome_names)
    ax_outcomes.set_xlabel('Contribution to Utility ($)', fontweight='bold', fontsize=12)
    ax_outcomes.set_title('Outcome-Specific Contributions (Green = Surgery Better, Red = Conservative Better)', 
                         fontweight='bold', fontsize=12)
    ax_outcomes.axvline(0, color='black', linewidth=2)
    ax_outcomes.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, outcome_values)):
        width = bar.get_width()
        label_x = width + (300 if width > 0 else -300)
        ax_outcomes.text(label_x, bar.get_y() + bar.get_height()/2, 
                        f'${val:,.0f}', ha='left' if width > 0 else 'right', 
                        va='center', fontweight='bold', fontsize=10)

    ax_sens = fig.add_subplot(gs[2, :])
    
    preferences = ['Pain', 'Function', 'Tx Pain', 'Quality', 'Cost']
    flippable = [0, 0, 0, 0, 0]
    
    for i in range(1, 6):
        key = f'WP{i}_increase'
        if key in sens['variations']:
            flippable[i-1] = 1 if sens['variations'][key]['changed'] else 0
    
    colors_sens = ['#ff6b6b' if f == 1 else '#95e1d3' for f in flippable]
    ax_sens.barh(preferences, flippable, color=colors_sens, edgecolor='black', linewidth=1.5)
    ax_sens.set_xlabel('Can Flip Decision (1=Yes, 0=No)', fontsize=11, fontweight='bold')
    ax_sens.set_title('Sensitivity Analysis: Which Preferences Can Change Decision?',
                     fontsize=12, fontweight='bold')
    ax_sens.set_xlim(0, 1.2)
    ax_sens.grid(axis='x', alpha=0.3)
    
    stability_color = '#d4edda' if sens['recommendation_stable'] else '#fff3cd'
    stability_text = f"Sensitivity Level:\n{sens['sensitivity_level']}\n\n"
    stability_text += f"Flippable: {sens['num_flippable']}/5\n"
    stability_text += f"Utility: ${abs(sens['base_utility']):,.0f}"
    
    ax_sens.text(1.05, 0.5, stability_text, transform=ax_sens.transAxes,
                fontsize=10, va='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor=stability_color, alpha=0.8, pad=1))

    ax_considerations = fig.add_subplot(gs[3, :])
    ax_considerations.axis('off')
    
    considerations_text = "KEY CLINICAL CONSIDERATIONS:\n\n"
    
    if not sens['recommendation_stable']:
        considerations_text += f"⚠️ SENSITIVITY WARNING: {sens['sensitivity_level']}\n"
        considerations_text += "Recommendation unstable at ±20% preference variation.\n\n"
    
    if rec['trade_offs']:
        considerations_text += "Trade-offs:\n"
        for i, trade_off in enumerate(rec['trade_offs'], 1):
            considerations_text += f"  {i}. {trade_off}\n"
        considerations_text += "\n"
    
    if rec['qualifications']:
        considerations_text += "Clinical Qualifications:\n"
        for i, qual in enumerate(rec['qualifications'], 1):
            considerations_text += f"  {i}. {qual}\n"
    
    if not rec['trade_offs'] and not rec['qualifications'] and sens['recommendation_stable']:
        considerations_text += "✓ Clear recommendation with stable preferences."
    
    ax_considerations.text(0.05, 0.95, considerations_text,
                          ha='left', va='top', fontsize=10,
                          family='monospace', wrap=True,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=1.5),
                          transform=ax_considerations.transAxes)
    
    plt.suptitle(f'Clinical Decision Support - Patient {patient_id}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig, rec, sens



def main():
    st.markdown('<p class="main-header">🏥 HAC Clinical Decision Support System</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    with st.sidebar:
        st.header("📊 Data Upload")
        
        # File uploads
        te_file = st.file_uploader(
            "Treatment Effects Data (CSV)",
            type=['csv'],
            help="Upload patient-specific treatment effects"
        )
        
        pref_file = st.file_uploader(
            "Patient Preferences Data (CSV)",
            type=['csv'],
            help="Upload patient preference weights"
        )
        
        st.markdown("---")
        st.header("⚙️ Configuration")
        
        n_clusters = st.slider(
            "Number of Clusters",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of patient clusters for HAC"
        )
        
        linkage_method = st.selectbox(
            "Linkage Method",
            options=['ward', 'complete', 'average', 'single'],
            index=0,
            help="Hierarchical clustering linkage method"
        )
        
        variation_pct = st.slider(
            "Sensitivity Variation %",
            min_value=10,
            max_value=30,
            value=20,
            help="Percentage variation for sensitivity analysis"
        ) / 100.0
        
        st.markdown("---")
        st.header("💡 Tips")
        st.info("""
        **Large Datasets (>10K patients):**
        - App uses optimized clustering
        - KMeans for speed
        - Sample dendrogram
        - All patients clustered
        
        **Performance:**
        - <1K: Instant
        - 1K-10K: ~5-10s
        - >10K: ~30-60s
        """)
    
    if te_file is None or pref_file is None:
        st.info("👈 Please upload both Treatment Effects and Patient Preferences data to begin")
        
        with st.expander("📋 Expected Data Format"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Treatment Effects CSV")
                st.code("""
PAT_ID,TE1,TE2,TE3,TE4,TE5
1,0.173,-0.206,0.334,19335.35,6.85
2,0.363,-0.344,0.295,24657.96,5.86
...
                """, language="csv")
            
            with col2:
                st.subheader("Patient Preferences CSV")
                st.code("""
PAT_ID,WP1,WP2,WP3,WP4,WP5
1,16.28,3.85,20.90,17.30,7.53
2,42.39,23.10,17.06,18.90,24.47
...
                """, language="csv")
        
        return
    
    try:
        df_te = pd.read_csv(te_file)
        df_pref = pd.read_csv(pref_file)
        
        st.success(f"✅ Loaded {len(df_te):,} patients from treatment effects data")
        st.success(f"✅ Loaded {len(df_pref):,} patients from preferences data")
        
        common_patients = set(df_te['PAT_ID']) & set(df_pref['PAT_ID'])
        
        if len(common_patients) < len(df_te) or len(common_patients) < len(df_pref):
            st.warning(f"""
            ⚠️ **Data Mismatch Detected**
            
            - Treatment Effects: {len(df_te):,} patients
            - Preferences: {len(df_pref):,} patients  
            - Common (usable): {len(common_patients):,} patients
            
            **Action:** Using only the {len(common_patients):,} patients present in both datasets.
            """)
            
            df_te = df_te[df_te['PAT_ID'].isin(common_patients)].reset_index(drop=True)
            df_pref = df_pref[df_pref['PAT_ID'].isin(common_patients)].reset_index(drop=True)
            
            st.info(f"📊 Proceeding with {len(df_te):,} matched patients")
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    dataset_size = len(df_te)
    
    if dataset_size > 10000:
        st.warning(f"""
        ⚠️ Large dataset detected ({dataset_size:,} patients)
        
        - Using optimized clustering (KMeans instead of HAC for speed)
        - Dendrogram will show sample of {min(5000, dataset_size)} patients
        - All patients will still be assigned to clusters
        - This may take 30-60 seconds...
        """)
    
    with st.spinner("Building clustering model..."):
        try:
            recommender = TreatmentRecommender(df_te, df_pref)
            recommender.build_clusters(n_clusters=n_clusters, linkage_method=linkage_method)
            
            with st.spinner("Computing cluster utilities..."):
                recommender.compute_cluster_utilities()
            
            sensitivity_analyzer = SensitivityAnalyzer(recommender)
            
            st.success(f"✅ Model initialized successfully! Processed {dataset_size:,} patients into {n_clusters} clusters")
            
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            st.error("Please try reducing the dataset size or check data format")
            return
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Patient Analysis",
        "📊 Cluster Overview",
        "🔍 Batch Analysis",
        "📈 Model Insights"
    ])
    

    with tab1:
        st.markdown('<p class="sub-header">Individual Patient Analysis</p>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            patient_id = st.number_input(
                "Enter Patient ID",
                min_value=int(df_te['PAT_ID'].min()),
                max_value=int(df_te['PAT_ID'].max()),
                value=int(df_te['PAT_ID'].min()),
                step=1
            )
        
        with col2:
            analyze_btn = st.button("🔬 Analyze Patient", type="primary", use_container_width=True)
        
        if analyze_btn:
            with st.spinner(f"Analyzing Patient {patient_id}..."):
                fig, rec, sens = create_comprehensive_dashboard(
                    patient_id, recommender, sensitivity_analyzer
                )
                
                if fig is None:
                    st.error("Patient not found in database")
                else:
                    st.pyplot(fig)
                    plt.close()
                    
                    st.markdown("---")
                    st.markdown("### 📋 Summary Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Recommendation",
                            rec['recommended_treatment'],
                            delta=rec['confidence']
                        )
                    
                    with col2:
                        st.metric(
                            "Utility Difference",
                            f"${rec['utility_difference']:,.0f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Stability",
                            "STABLE" if sens['recommendation_stable'] else "UNSTABLE",
                            delta=f"{sens['num_flippable']}/5 flippable"
                        )
                    
                    with col4:
                        st.metric(
                            "Sensitivity",
                            sens['sensitivity_level'].split('(')[0].strip()
                        )
                    
                    with st.expander("📊 Detailed Results"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Scaled Outcome Values:**")
                            scaled = rec['utility_details']['scaled_outcomes']
                            st.write(f"Pain Improvement: ${scaled['pain_improvement_value']:,.0f}")
                            st.write(f"Function Improvement: ${scaled['function_improvement_value']:,.0f}")
                            st.write(f"Treatment Pain Cost: ${scaled['treatment_pain_cost']:,.0f}")
                            st.write(f"Quality Value: ${scaled['additional_quality_value']:,.0f}")
                            st.write(f"Direct Cost: ${scaled['direct_cost']:,.0f}")
                        
                        with col2:
                            st.markdown("**Weighted Contributions:**")
                            comps = rec['utility_details']['components']
                            st.write(f"Pain: ${comps['pain_improvement_value']['contribution']:,.0f}")
                            st.write(f"Function: ${comps['function_improvement_value']['contribution']:,.0f}")
                            st.write(f"Tx Pain: ${comps['treatment_pain_cost']['contribution']:,.0f}")
                            st.write(f"Quality: ${comps['additional_quality_value']['contribution']:,.0f}")
                            st.write(f"Cost: ${comps['direct_cost']['contribution']:,.0f}")
                    
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download Report",
                        data=buffer,
                        file_name=f"patient_{patient_id}_cds_report.png",
                        mime="image/png"
                    )
    

    with tab2:
        st.markdown('<p class="sub-header">Cluster Analysis</p>', 
                   unsafe_allow_html=True)
        
        # Dendrogram
        st.markdown("### 🌳 Clustering Dendrogram")
        
        if len(df_te) > 5000:
            st.info(f"📊 Showing dendrogram for sample of 5,000 patients (total: {len(df_te):,}). All patients are still clustered.")
        
        fig_dend = plot_dendrogram(
            recommender.linkage_matrix, 
            title=f"Hierarchical Clustering Dendrogram ({min(5000, len(df_te))} patients)"
        )
        st.pyplot(fig_dend)
        plt.close()
        
        st.markdown("### 📊 Cluster Statistics")
        
        cluster_stats = []
        for cluster_id, stats in recommender.cluster_utilities.items():
            cluster_stats.append({
                'Cluster': cluster_id,
                'Mean Utility': f"${stats['mean']:,.2f}",
                'Std Utility': f"${stats['std']:,.2f}",
                'Patient Count': stats['count']
            })
        
        df_cluster_stats = pd.DataFrame(cluster_stats)
        st.dataframe(df_cluster_stats, use_container_width=True)
        
        st.markdown("### 📈 Cluster Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        cluster_counts = recommender.te_df['cluster'].value_counts().sort_index()
        ax.bar(cluster_counts.index, cluster_counts.values, color='steelblue', alpha=0.7)
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Number of Patients', fontsize=12)
        ax.set_title('Patient Distribution Across Clusters', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        st.markdown('<p class="sub-header">Batch Patient Analysis</p>', 
                   unsafe_allow_html=True)
        
        num_patients = st.slider(
            "Number of patients to analyze",
            min_value=5,
            max_value=min(100, len(df_te)),
            value=20
        )
        
        if st.button("🚀 Run Batch Analysis", type="primary"):
            with st.spinner("Running batch analysis..."):
                sample_ids = df_te['PAT_ID'].sample(num_patients, random_state=42).values
                
                results = []
                progress_bar = st.progress(0)
                
                for idx, pid in enumerate(sample_ids):
                    try:
                        rec = recommender.recommend_for_patient(int(pid))
                        sens = sensitivity_analyzer.analyze(int(pid), variation_pct=variation_pct)
                        
                        if 'error' not in rec:
                            results.append({
                                'Patient_ID': pid,
                                'Recommendation': rec['recommended_treatment'],
                                'Confidence': rec['confidence'],
                                'Utility_Diff': rec['utility_difference'],
                                'Stability': 'Stable' if sens['recommendation_stable'] else 'Unstable',
                                'Sensitivity_Level': sens['sensitivity_level'],
                                'Flippable_Prefs': sens['num_flippable'],
                                'Cluster': rec.get('cluster', 'N/A')
                            })
                    except Exception as e:
                        st.warning(f"Skipping patient {pid}: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / num_patients)
                
                df_results = pd.DataFrame(results)
                
                st.markdown("### 📊 Batch Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    surgery_pct = (df_results['Recommendation'] == 'Surgery').mean() * 100
                    st.metric("Surgery %", f"{surgery_pct:.1f}%")
                
                with col2:
                    stable_pct = (df_results['Stability'] == 'Stable').mean() * 100
                    st.metric("Stable %", f"{stable_pct:.1f}%")
                
                with col3:
                    high_conf_pct = (df_results['Confidence'] == 'High').mean() * 100
                    st.metric("High Confidence %", f"{high_conf_pct:.1f}%")
                
                with col4:
                    avg_flip = df_results['Flippable_Prefs'].mean()
                    st.metric("Avg Flippable", f"{avg_flip:.2f}")
                
                st.markdown("### 📋 Detailed Results")
                st.dataframe(df_results, use_container_width=True)
                
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="batch_analysis_results.csv",
                    mime="text/csv"
                )
    
    with tab4:
        st.markdown('<p class="sub-header">Model Insights & Diagnostics</p>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Data Overview")
            st.write(f"**Total Patients:** {len(df_te):,}")
            st.write(f"**Number of Clusters:** {n_clusters}")
            st.write(f"**Linkage Method:** {linkage_method}")
            st.write(f"**Sensitivity Variation:** ±{variation_pct*100:.0f}%")
            
            st.markdown("### 🔍 Treatment Effects Distribution")
            te_display_cols = []
            for col in df_te.columns:
                if col.startswith('TE'):
                    te_display_cols.append(col)
            
            if te_display_cols:
                te_stats = df_te[te_display_cols].describe()
                st.dataframe(te_stats, use_container_width=True)
            else:
                st.info("No treatment effect columns found")
        
        with col2:
            st.markdown("### 📈 Preference Weights Distribution")
            pref_cols = [c for c in df_pref.columns if c.startswith('WP') and not c.endswith('_normalized')]
            if pref_cols:
                pref_stats = df_pref[pref_cols].describe()
                st.dataframe(pref_stats, use_container_width=True)
            else:
                st.info("No preference columns found")
        
        st.markdown("### 🔗 Treatment Effects Correlation")
        te_display_cols = [c for c in df_te.columns if c.startswith('TE')]
        
        if len(te_display_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df_te[te_display_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title('Treatment Effects Correlation Matrix', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Not enough treatment effect columns for correlation analysis")


if __name__ == "__main__":
    main()