#!/usr/bin/env python3
"""
Demo script for Vietnamese NL2SQL Statistical Framework
Shows the statistical rigor implementation with sample data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from datetime import datetime
import os

# Set style for publication-ready charts
plt.style.use('default')
sns.set_palette("husl")

def create_demo_data():
    """Create realistic demo data for 150 Vietnamese queries"""
    np.random.seed(42)
    
    # Stratified sampling: 50 queries per complexity level
    complexities = ['Simple'] * 50 + ['Medium'] * 50 + ['Complex'] * 50
    
    data = []
    
    for i, complexity in enumerate(complexities):
        # Simulate realistic performance differences based on your research
        if complexity == 'Simple':
            p1_ex_prob, p2_ex_prob = 0.85, 0.80  # Pipeline 1 advantage
            p1_em_prob, p2_em_prob = 0.60, 0.50
            p1_latency_mean, p2_latency_mean = 600, 900  # Pipeline 1 faster
        elif complexity == 'Medium':
            p1_ex_prob, p2_ex_prob = 0.70, 0.65
            p1_em_prob, p2_em_prob = 0.40, 0.32
            p1_latency_mean, p2_latency_mean = 850, 1250
        else:  # Complex
            p1_ex_prob, p2_ex_prob = 0.60, 0.55
            p1_em_prob, p2_em_prob = 0.25, 0.18
            p1_latency_mean, p2_latency_mean = 1200, 1800
        
        # Generate binary outcomes
        p1_ex = np.random.binomial(1, p1_ex_prob)
        p2_ex = np.random.binomial(1, p2_ex_prob)
        p1_em = np.random.binomial(1, p1_em_prob)
        p2_em = np.random.binomial(1, p2_em_prob)
        
        # Generate latency (average of 3 replicates as per statistical rigor)
        p1_latencies = np.random.normal(p1_latency_mean, p1_latency_mean * 0.2, 3)
        p2_latencies = np.random.normal(p2_latency_mean, p2_latency_mean * 0.2, 3)
        
        p1_latency = np.mean(p1_latencies)
        p2_latency = np.mean(p2_latencies)
        
        # Generate GPU metrics (Pipeline 1 more efficient)
        p1_gpu = np.random.normal(4.2, 0.5)
        p2_gpu = np.random.normal(6.8, 0.8)
        
        # Generate error counts (Pipeline 1 fewer errors)
        p1_tonal = np.random.poisson(0.1 if complexity == 'Simple' else 0.3)
        p2_tonal = np.random.poisson(0.15 if complexity == 'Simple' else 0.4)
        
        data.append({
            'query_id': f'Q{i+1:03d}',
            'complexity': complexity,
            'vietnamese_query': f'Vietnamese query {i+1}',
            'pipeline1_execution_correct': p1_ex,
            'pipeline2_execution_correct': p2_ex,
            'pipeline1_exact_match': p1_em,
            'pipeline2_exact_match': p2_em,
            'pipeline1_latency_ms': p1_latency,
            'pipeline2_latency_ms': p2_latency,
            'pipeline1_gpu_memory_gb': p1_gpu,
            'pipeline2_gpu_memory_gb': p2_gpu,
            'pipeline1_tonal_accent_errors': p1_tonal,
            'pipeline2_tonal_accent_errors': p2_tonal,
            'pipeline1_compound_word_errors': np.random.poisson(0.2),
            'pipeline2_compound_word_errors': np.random.poisson(0.3),
            'pipeline1_sql_syntax_errors': np.random.poisson(0.1),
            'pipeline2_sql_syntax_errors': np.random.poisson(0.15),
            'pipeline1_schema_logic_errors': np.random.poisson(0.05),
            'pipeline2_schema_logic_errors': np.random.poisson(0.08)
        })
    
    return pd.DataFrame(data)

def wilson_confidence_interval(successes, total, confidence=0.95):
    """Calculate Wilson confidence interval for proportions"""
    if total == 0:
        return (0.0, 0.0)
        
    z = stats.norm.ppf((1 + confidence) / 2)
    p = successes / total
    n = total
    
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    
    return (max(0, centre - half_width), min(1, centre + half_width))

def mcnemar_test(pipeline1_correct, pipeline2_correct):
    """Perform McNemar's test for paired binary outcomes"""
    # Create contingency table
    both_correct = np.sum((pipeline1_correct == 1) & (pipeline2_correct == 1))
    p1_only = np.sum((pipeline1_correct == 1) & (pipeline2_correct == 0))
    p2_only = np.sum((pipeline1_correct == 0) & (pipeline2_correct == 1))
    both_wrong = np.sum((pipeline1_correct == 0) & (pipeline2_correct == 0))
    
    # McNemar's test focuses on discordant pairs
    if p1_only + p2_only == 0:
        return {'statistic': 0.0, 'p_value': 1.0}
    
    # McNemar's chi-square statistic
    chi2_stat = (abs(p1_only - p2_only) - 1)**2 / (p1_only + p2_only)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
    
    return {'statistic': chi2_stat, 'p_value': p_value}

def create_accuracy_vs_complexity_chart(df, output_dir):
    """Generate accuracy vs query complexity chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    complexities = ['Simple', 'Medium', 'Complex']
    colors = {'pipeline1': '#2E86AB', 'pipeline2': '#A23B72'}
    
    # Execution Accuracy Chart
    p1_ex_rates = []
    p1_ex_cis = []
    p2_ex_rates = []
    p2_ex_cis = []
    
    for complexity in complexities:
        subset = df[df['complexity'] == complexity]
        
        p1_successes = subset['pipeline1_execution_correct'].sum()
        p2_successes = subset['pipeline2_execution_correct'].sum()
        n = len(subset)
        
        p1_rate = p1_successes / n
        p2_rate = p2_successes / n
        
        p1_ci = wilson_confidence_interval(p1_successes, n)
        p2_ci = wilson_confidence_interval(p2_successes, n)
        
        p1_ex_rates.append(p1_rate * 100)
        p1_ex_cis.append([
            (p1_rate - p1_ci[0]) * 100,
            (p1_ci[1] - p1_rate) * 100
        ])
        p2_ex_rates.append(p2_rate * 100)
        p2_ex_cis.append([
            (p2_rate - p2_ci[0]) * 100,
            (p2_ci[1] - p2_rate) * 100
        ])
    
    x = np.arange(len(complexities))
    width = 0.35
    
    p1_ex_cis = np.array(p1_ex_cis).T
    p2_ex_cis = np.array(p2_ex_cis).T
    
    bars1 = ax1.bar(x - width/2, p1_ex_rates, width, 
                   yerr=p1_ex_cis, capsize=5,
                   label='Pipeline 1 (End-to-End Vietnamese)', 
                   color=colors['pipeline1'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, p2_ex_rates, width,
                   yerr=p2_ex_cis, capsize=5,
                   label='Pipeline 2 (Hybrid Translation)', 
                   color=colors['pipeline2'], alpha=0.8)
    
    ax1.set_xlabel('Query Complexity', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Execution Accuracy vs Query Complexity\n(with 95% Wilson Confidence Intervals)', 
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(complexities)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + 2,
                f'{height1:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax1.text(bar2.get_x() + bar2.get_width()/2., height2 + 2,
                f'{height2:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Exact Match Chart (similar structure)
    p1_em_rates = []
    p1_em_cis = []
    p2_em_rates = []
    p2_em_cis = []
    
    for complexity in complexities:
        subset = df[df['complexity'] == complexity]
        
        p1_successes = subset['pipeline1_exact_match'].sum()
        p2_successes = subset['pipeline2_exact_match'].sum()
        n = len(subset)
        
        p1_rate = p1_successes / n
        p2_rate = p2_successes / n
        
        p1_ci = wilson_confidence_interval(p1_successes, n)
        p2_ci = wilson_confidence_interval(p2_successes, n)
        
        p1_em_rates.append(p1_rate * 100)
        p1_em_cis.append([
            (p1_rate - p1_ci[0]) * 100,
            (p1_ci[1] - p1_rate) * 100
        ])
        p2_em_rates.append(p2_rate * 100)
        p2_em_cis.append([
            (p2_rate - p2_ci[0]) * 100,
            (p2_ci[1] - p2_rate) * 100
        ])
    
    p1_em_cis = np.array(p1_em_cis).T
    p2_em_cis = np.array(p2_em_cis).T
    
    bars3 = ax2.bar(x - width/2, p1_em_rates, width,
                   yerr=p1_em_cis, capsize=5,
                   label='Pipeline 1 (End-to-End Vietnamese)', 
                   color=colors['pipeline1'], alpha=0.8)
    bars4 = ax2.bar(x + width/2, p2_em_rates, width,
                   yerr=p2_em_cis, capsize=5,
                   label='Pipeline 2 (Hybrid Translation)', 
                   color=colors['pipeline2'], alpha=0.8)
    
    ax2.set_xlabel('Query Complexity', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Exact Match (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Exact Match vs Query Complexity\n(with 95% Wilson Confidence Intervals)', 
                 fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(complexities)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Add value labels
    for i, (bar3, bar4) in enumerate(zip(bars3, bars4)):
        height3 = bar3.get_height()
        height4 = bar4.get_height()
        ax2.text(bar3.get_x() + bar3.get_width()/2., height3 + 1,
                f'{height3:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax2.text(bar4.get_x() + bar4.get_width()/2., height4 + 1,
                f'{height4:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    chart_path = f"{output_dir}/accuracy_vs_complexity.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return chart_path

def create_error_breakdown_chart(df, output_dir):
    """Generate error type breakdown chart"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate error totals
    error_types = ['tonal_accent_errors', 'compound_word_errors', 
                  'sql_syntax_errors', 'schema_logic_errors']
    error_labels = ['Tonal/Accent\nErrors', 'Compound Word\nErrors', 
                   'SQL Syntax\nErrors', 'Schema Logic\nErrors']
    
    p1_totals = []
    p2_totals = []
    
    for error_type in error_types:
        p1_col = f'pipeline1_{error_type}'
        p2_col = f'pipeline2_{error_type}'
        
        p1_totals.append(df[p1_col].sum())
        p2_totals.append(df[p2_col].sum())
    
    # Calculate percentages
    p1_sum = sum(p1_totals)
    p2_sum = sum(p2_totals)
    
    p1_percentages = [x/p1_sum * 100 for x in p1_totals] if p1_sum > 0 else [0] * len(p1_totals)
    p2_percentages = [x/p2_sum * 100 for x in p2_totals] if p2_sum > 0 else [0] * len(p2_totals)
    
    # Create stacked bar chart
    x = ['Pipeline 1\n(End-to-End Vietnamese)', 'Pipeline 2\n(Hybrid Translation)']
    
    # Colors for different error types
    error_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    bottom_p1 = 0
    bottom_p2 = 0
    
    for i, (error_label, color) in enumerate(zip(error_labels, error_colors)):
        p1_height = p1_percentages[i]
        p2_height = p2_percentages[i]
        
        ax.bar(x[0], p1_height, bottom=bottom_p1, color=color, 
              alpha=0.8, label=error_label if i == 0 else "")
        ax.bar(x[1], p2_height, bottom=bottom_p2, color=color, alpha=0.8)
        
        # Add percentage labels if significant
        if p1_height > 5:
            ax.text(0, bottom_p1 + p1_height/2, f'{p1_height:.1f}%\n({p1_totals[i]})', 
                   ha='center', va='center', fontweight='bold', fontsize=10)
        if p2_height > 5:
            ax.text(1, bottom_p2 + p2_height/2, f'{p2_height:.1f}%\n({p2_totals[i]})', 
                   ha='center', va='center', fontweight='bold', fontsize=10)
        
        bottom_p1 += p1_height
        bottom_p2 += p2_height
    
    # Create custom legend
    import matplotlib.patches as mpatches
    legend_elements = [mpatches.Patch(color=color, label=label) 
                      for color, label in zip(error_colors, error_labels)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    ax.set_ylabel('Error Distribution (%)', fontsize=12, fontweight='bold')
    ax.set_title('Error Type Breakdown by Pipeline\n(Stacked 100% Distribution)', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add total error counts at the bottom
    ax.text(0, -8, f'Total Errors: {p1_sum}', ha='center', va='top', 
           fontweight='bold', fontsize=11)
    ax.text(1, -8, f'Total Errors: {p2_sum}', ha='center', va='top', 
           fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    
    chart_path = f"{output_dir}/error_type_breakdown.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return chart_path

def run_statistical_analysis(df):
    """Run comprehensive statistical analysis"""
    print("Running Statistical Analysis...")
    print("=" * 50)
    
    # Overall Execution Accuracy Analysis
    p1_ex_all = df['pipeline1_execution_correct'].values
    p2_ex_all = df['pipeline2_execution_correct'].values
    
    # Point estimates with Wilson CIs
    n_total = len(df)
    p1_ex_successes = np.sum(p1_ex_all)
    p2_ex_successes = np.sum(p2_ex_all)
    
    p1_ex_rate = p1_ex_successes / n_total
    p2_ex_rate = p2_ex_successes / n_total
    
    p1_ex_ci = wilson_confidence_interval(p1_ex_successes, n_total)
    p2_ex_ci = wilson_confidence_interval(p2_ex_successes, n_total)
    
    # McNemar's test
    mcnemar_result = mcnemar_test(p1_ex_all, p2_ex_all)
    
    # Risk difference
    risk_diff = p1_ex_rate - p2_ex_rate
    
    # Report template (as specified in requirements)
    ex_report = f"Pipeline-1 EX = {p1_ex_rate:.1%} (95% CI: {p1_ex_ci[0]:.1%}–{p1_ex_ci[1]:.1%}), " \
               f"Pipeline-2 EX = {p2_ex_rate:.1%} (95% CI: {p2_ex_ci[0]:.1%}–{p2_ex_ci[1]:.1%}); " \
               f"McNemar p={mcnemar_result['p_value']:.3f}; " \
               f"ΔEX = {risk_diff:.1%}"
    
    print("EXECUTION ACCURACY RESULTS:")
    print(ex_report)
    print()
    
    # Latency Analysis
    p1_latency = df['pipeline1_latency_ms'].values
    p2_latency = df['pipeline2_latency_ms'].values
    differences = p1_latency - p2_latency
    
    # Test for normality
    shapiro_stat, shapiro_p = stats.shapiro(differences)
    is_normal = shapiro_p > 0.05
    
    if is_normal:
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(p1_latency, p2_latency)
        mean_diff = np.mean(differences)
        print(f"LATENCY ANALYSIS (Paired t-test):")
        print(f"Mean difference: {mean_diff:.1f} ms")
        print(f"t-statistic: {t_stat:.3f}")
        print(f"p-value: {p_value:.3f}")
    else:
        # Wilcoxon signed-rank test
        w_stat, p_value = stats.wilcoxon(p1_latency, p2_latency)
        median_diff = np.median(differences)
        print(f"LATENCY ANALYSIS (Wilcoxon signed-rank):")
        print(f"Median difference: {median_diff:.1f} ms")
        print(f"W-statistic: {w_stat:.3f}")
        print(f"p-value: {p_value:.3f}")
    
    print()
    
    # Complexity Breakdown
    print("COMPLEXITY BREAKDOWN:")
    for complexity in ['Simple', 'Medium', 'Complex']:
        subset = df[df['complexity'] == complexity]
        if len(subset) > 0:
            p1_rate = subset['pipeline1_execution_correct'].mean()
            p2_rate = subset['pipeline2_execution_correct'].mean()
            print(f"{complexity}: P1={p1_rate:.1%}, P2={p2_rate:.1%}")
    
    return {
        'execution_accuracy_report': ex_report,
        'latency_analysis': {
            'test_type': 'paired_t_test' if is_normal else 'wilcoxon_signed_rank',
            'p_value': p_value,
            'mean_difference': np.mean(differences),
            'median_difference': np.median(differences)
        },
        'sample_size': n_total
    }

def main():
    """Main demonstration function"""
    print("Vietnamese NL2SQL Statistical Framework Demo")
    print("=" * 50)
    print("Implementing Statistical Rigor as per Requirements:")
    print("- 150 Vietnamese queries (50 Simple, 50 Medium, 50 Complex)")
    print("- Paired measurements per query")
    print("- Wilson 95% CIs for proportions")
    print("- McNemar's test for binary outcomes")
    print("- Paired t-test/Wilcoxon for continuous measures")
    print()
    
    # Create output directory
    output_dir = "statistical_demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate demo data
    print("Generating demo data...")
    df = create_demo_data()
    print(f"Created dataset: {len(df)} queries")
    print(f"Complexity distribution: {df['complexity'].value_counts().to_dict()}")
    print()
    
    # Run statistical analysis
    analysis_results = run_statistical_analysis(df)
    
    # Generate charts
    print("Generating statistical visualization charts...")
    
    chart1 = create_accuracy_vs_complexity_chart(df, output_dir)
    print(f"✓ Accuracy vs Complexity chart: {chart1}")
    
    chart2 = create_error_breakdown_chart(df, output_dir)
    print(f"✓ Error Type Breakdown chart: {chart2}")
    
    # Save results
    results_file = f"{output_dir}/statistical_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'analysis_results': analysis_results,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'sample_size': len(df),
                'methodology': 'Paired measurements with statistical rigor'
            }
        }, f, indent=2, default=str)
    
    print(f"✓ Statistical results saved: {results_file}")
    print()
    print("DEMONSTRATION COMPLETED!")
    print(f"All outputs saved to: {output_dir}/")
    print()
    print("Key Statistical Findings:")
    print(f"- {analysis_results['execution_accuracy_report']}")
    print(f"- Latency test: {analysis_results['latency_analysis']['test_type']}")
    print(f"- Sample size: {analysis_results['sample_size']} queries")

if __name__ == "__main__":
    main()
