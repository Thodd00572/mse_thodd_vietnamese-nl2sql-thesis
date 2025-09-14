"""
Statistical Analysis Framework for Vietnamese NL2SQL Pipeline Evaluation
Implements rigorous statistical methods for paired pipeline comparison
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import wilcoxon, shapiro, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests
import json
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class VietnameseNL2SQLStatistics:
    """
    Statistical analysis framework for Vietnamese NL2SQL pipeline evaluation
    Implements paired measurements with proper statistical rigor
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize statistical analysis framework"""
        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.results = {}
        
    def wilson_confidence_interval(self, successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Wilson confidence interval for proportions
        More accurate than normal approximation for small samples
        """
        if total == 0:
            return (0.0, 0.0)
            
        z = stats.norm.ppf((1 + confidence) / 2)
        p = successes / total
        n = total
        
        denominator = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denominator
        half_width = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
        
        return (max(0, centre - half_width), min(1, centre + half_width))
    
    def mcnemar_test(self, pipeline1_correct: np.array, pipeline2_correct: np.array) -> Dict:
        """
        Perform McNemar's test for paired binary outcomes
        Tests if there's a significant difference between paired proportions
        """
        # Create contingency table
        both_correct = np.sum((pipeline1_correct == 1) & (pipeline2_correct == 1))
        p1_only = np.sum((pipeline1_correct == 1) & (pipeline2_correct == 0))
        p2_only = np.sum((pipeline1_correct == 0) & (pipeline2_correct == 1))
        both_wrong = np.sum((pipeline1_correct == 0) & (pipeline2_correct == 0))
        
        # McNemar's test focuses on discordant pairs
        if p1_only + p2_only == 0:
            return {
                'statistic': 0.0,
                'p_value': 1.0,
                'contingency_table': [[both_correct, p1_only], [p2_only, both_wrong]],
                'interpretation': 'No discordant pairs - pipelines perform identically'
            }
        
        # McNemar's chi-square statistic
        chi2_stat = (abs(p1_only - p2_only) - 1)**2 / (p1_only + p2_only)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
        
        return {
            'statistic': chi2_stat,
            'p_value': p_value,
            'contingency_table': [[both_correct, p1_only], [p2_only, both_wrong]],
            'discordant_pairs': p1_only + p2_only,
            'p1_advantage': p1_only,
            'p2_advantage': p2_only
        }
    
    def calculate_effect_sizes(self, pipeline1_correct: np.array, pipeline2_correct: np.array) -> Dict:
        """Calculate effect sizes for binary outcomes"""
        p1_rate = np.mean(pipeline1_correct)
        p2_rate = np.mean(pipeline2_correct)
        
        # Risk difference (difference in proportions)
        risk_diff = p1_rate - p2_rate
        
        # Odds ratio calculation
        p1_odds = p1_rate / (1 - p1_rate) if p1_rate < 1 else float('inf')
        p2_odds = p2_rate / (1 - p2_rate) if p2_rate < 1 else float('inf')
        odds_ratio = p1_odds / p2_odds if p2_odds > 0 else float('inf')
        
        # Bootstrap confidence intervals for risk difference
        n_bootstrap = 1000
        bootstrap_diffs = []
        n = len(pipeline1_correct)
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            p1_boot = np.mean(pipeline1_correct[idx])
            p2_boot = np.mean(pipeline2_correct[idx])
            bootstrap_diffs.append(p1_boot - p2_boot)
        
        risk_diff_ci = np.percentile(bootstrap_diffs, [2.5, 97.5])
        
        return {
            'risk_difference': risk_diff,
            'risk_difference_ci': risk_diff_ci,
            'odds_ratio': odds_ratio,
            'pipeline1_rate': p1_rate,
            'pipeline2_rate': p2_rate
        }
    
    def analyze_accuracy_metrics(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze accuracy metrics (EX and EM) with statistical rigor
        Implements paired comparison with Wilson CIs and McNemar's test
        """
        analysis = {}
        
        # Analyze by complexity strata
        complexities = ['Simple', 'Medium', 'Complex']
        
        for complexity in complexities:
            if complexity in results_df['complexity'].values:
                subset = results_df[results_df['complexity'] == complexity]
                
                # Extract binary outcomes
                p1_ex = subset['pipeline1_execution_correct'].values
                p2_ex = subset['pipeline2_execution_correct'].values
                p1_em = subset['pipeline1_exact_match'].values
                p2_em = subset['pipeline2_exact_match'].values
                
                complexity_analysis = {}
                
                # Execution Accuracy (EX) Analysis
                ex_analysis = self._analyze_binary_metric(p1_ex, p2_ex, 'Execution Accuracy')
                complexity_analysis['execution_accuracy'] = ex_analysis
                
                # Exact Match (EM) Analysis
                em_analysis = self._analyze_binary_metric(p1_em, p2_em, 'Exact Match')
                complexity_analysis['exact_match'] = em_analysis
                
                analysis[complexity] = complexity_analysis
        
        # Overall analysis (all complexities combined)
        p1_ex_all = results_df['pipeline1_execution_correct'].values
        p2_ex_all = results_df['pipeline2_execution_correct'].values
        p1_em_all = results_df['pipeline1_exact_match'].values
        p2_em_all = results_df['pipeline2_exact_match'].values
        
        overall_analysis = {}
        overall_analysis['execution_accuracy'] = self._analyze_binary_metric(p1_ex_all, p2_ex_all, 'Execution Accuracy')
        overall_analysis['exact_match'] = self._analyze_binary_metric(p1_em_all, p2_em_all, 'Exact Match')
        
        analysis['Overall'] = overall_analysis
        
        return analysis
    
    def _analyze_binary_metric(self, pipeline1_results: np.array, pipeline2_results: np.array, metric_name: str) -> Dict:
        """Helper method to analyze binary metrics"""
        n = len(pipeline1_results)
        
        # Point estimates with Wilson CIs
        p1_successes = np.sum(pipeline1_results)
        p2_successes = np.sum(pipeline2_results)
        
        p1_rate = p1_successes / n
        p2_rate = p2_successes / n
        
        p1_ci = self.wilson_confidence_interval(p1_successes, n)
        p2_ci = self.wilson_confidence_interval(p2_successes, n)
        
        # McNemar's test
        mcnemar_result = self.mcnemar_test(pipeline1_results, pipeline2_results)
        
        # Effect sizes
        effect_sizes = self.calculate_effect_sizes(pipeline1_results, pipeline2_results)
        
        # Generate report template
        report = f"Pipeline-1 {metric_name} = {p1_rate:.1%} (95% CI: {p1_ci[0]:.1%}–{p1_ci[1]:.1%}), " \
                f"Pipeline-2 {metric_name} = {p2_rate:.1%} (95% CI: {p2_ci[0]:.1%}–{p2_ci[1]:.1%}); " \
                f"McNemar p={mcnemar_result['p_value']:.3f}; " \
                f"Δ{metric_name} = {effect_sizes['risk_difference']:.1%} " \
                f"(95% CI: {effect_sizes['risk_difference_ci'][0]:.1%}–{effect_sizes['risk_difference_ci'][1]:.1%})."
        
        return {
            'pipeline1_rate': p1_rate,
            'pipeline1_ci': p1_ci,
            'pipeline2_rate': p2_rate,
            'pipeline2_ci': p2_ci,
            'mcnemar_test': mcnemar_result,
            'effect_sizes': effect_sizes,
            'sample_size': n,
            'report_template': report
        }
    
    def analyze_latency_metrics(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze latency metrics with proper statistical tests
        Uses paired t-test or Wilcoxon signed-rank based on normality
        """
        analysis = {}
        complexities = ['Simple', 'Medium', 'Complex', 'Overall']
        
        for complexity in complexities:
            if complexity == 'Overall':
                subset = results_df
            else:
                subset = results_df[results_df['complexity'] == complexity]
            
            if len(subset) == 0:
                continue
                
            # Extract latency measurements (average of 3 replicates)
            p1_latency = subset['pipeline1_latency_ms'].values
            p2_latency = subset['pipeline2_latency_ms'].values
            
            # Calculate paired differences
            differences = p1_latency - p2_latency
            
            # Test for normality of differences
            if len(differences) >= 3:
                shapiro_stat, shapiro_p = shapiro(differences)
                is_normal = shapiro_p > 0.05
            else:
                is_normal = True  # Assume normal for very small samples
            
            # Choose appropriate test
            if is_normal:
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(p1_latency, p2_latency)
                mean_diff = np.mean(differences)
                std_diff = np.std(differences, ddof=1)
                se_diff = std_diff / np.sqrt(len(differences))
                
                # 95% CI for mean difference
                t_critical = stats.t.ppf(0.975, len(differences) - 1)
                ci_lower = mean_diff - t_critical * se_diff
                ci_upper = mean_diff + t_critical * se_diff
                
                # Cohen's d (paired)
                cohens_d = mean_diff / std_diff
                
                test_result = {
                    'test_type': 'paired_t_test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'mean_difference': mean_diff,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'effect_size': cohens_d,
                    'effect_size_name': 'Cohen\'s d'
                }
            else:
                # Wilcoxon signed-rank test
                w_stat, p_value = wilcoxon(p1_latency, p2_latency)
                median_diff = np.median(differences)
                
                # Bootstrap CI for median difference
                n_bootstrap = 1000
                bootstrap_medians = []
                n = len(differences)
                
                for _ in range(n_bootstrap):
                    idx = np.random.choice(n, n, replace=True)
                    bootstrap_medians.append(np.median(differences[idx]))
                
                ci_lower, ci_upper = np.percentile(bootstrap_medians, [2.5, 97.5])
                
                # Cliff's delta (non-parametric effect size)
                cliffs_delta = self._calculate_cliffs_delta(p1_latency, p2_latency)
                
                test_result = {
                    'test_type': 'wilcoxon_signed_rank',
                    'statistic': w_stat,
                    'p_value': p_value,
                    'median_difference': median_diff,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'effect_size': cliffs_delta,
                    'effect_size_name': 'Cliff\'s delta'
                }
            
            # Descriptive statistics
            descriptive = {
                'pipeline1_mean': np.mean(p1_latency),
                'pipeline1_median': np.median(p1_latency),
                'pipeline1_std': np.std(p1_latency, ddof=1),
                'pipeline1_mad': stats.median_abs_deviation(p1_latency),
                'pipeline2_mean': np.mean(p2_latency),
                'pipeline2_median': np.median(p2_latency),
                'pipeline2_std': np.std(p2_latency, ddof=1),
                'pipeline2_mad': stats.median_abs_deviation(p2_latency),
                'sample_size': len(p1_latency),
                'normality_test': {
                    'shapiro_statistic': shapiro_stat if len(differences) >= 3 else None,
                    'shapiro_p_value': shapiro_p if len(differences) >= 3 else None,
                    'is_normal': is_normal
                }
            }
            
            analysis[complexity] = {
                'test_result': test_result,
                'descriptive_stats': descriptive
            }
        
        return analysis
    
    def _calculate_cliffs_delta(self, x: np.array, y: np.array) -> float:
        """Calculate Cliff's delta effect size for non-parametric comparison"""
        n_x, n_y = len(x), len(y)
        dominance = 0
        
        for xi in x:
            for yi in y:
                if xi > yi:
                    dominance += 1
                elif xi < yi:
                    dominance -= 1
        
        return dominance / (n_x * n_y)
    
    def analyze_gpu_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Analyze GPU usage metrics (peak GB and GPU-seconds)"""
        analysis = {}
        
        # GPU Peak Memory Analysis
        gpu_memory_analysis = self.analyze_latency_metrics(
            results_df.rename(columns={
                'pipeline1_gpu_memory_gb': 'pipeline1_latency_ms',
                'pipeline2_gpu_memory_gb': 'pipeline2_latency_ms'
            })
        )
        
        # GPU-seconds Analysis (if available)
        if 'pipeline1_gpu_seconds' in results_df.columns:
            gpu_seconds_analysis = self.analyze_latency_metrics(
                results_df.rename(columns={
                    'pipeline1_gpu_seconds': 'pipeline1_latency_ms',
                    'pipeline2_gpu_seconds': 'pipeline2_latency_ms'
                })
            )
            analysis['gpu_seconds'] = gpu_seconds_analysis
        
        analysis['gpu_memory'] = gpu_memory_analysis
        return analysis
    
    def analyze_error_typology(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze error types with statistical comparison
        Uses McNemar's test for binary comparisons and Bowker's test for multi-category
        """
        analysis = {}
        
        # Error type categories
        error_types = ['tonal_accent_errors', 'compound_word_errors', 
                      'sql_syntax_errors', 'schema_logic_errors']
        
        for error_type in error_types:
            p1_col = f'pipeline1_{error_type}'
            p2_col = f'pipeline2_{error_type}'
            
            if p1_col in results_df.columns and p2_col in results_df.columns:
                p1_errors = results_df[p1_col].values
                p2_errors = results_df[p2_col].values
                
                # Convert to binary (error occurred or not)
                p1_binary = (p1_errors > 0).astype(int)
                p2_binary = (p2_errors > 0).astype(int)
                
                # McNemar's test for this error type
                mcnemar_result = self.mcnemar_test(p1_binary, p2_binary)
                
                # Proportions with Wilson CIs
                p1_rate = np.mean(p1_binary)
                p2_rate = np.mean(p2_binary)
                p1_ci = self.wilson_confidence_interval(np.sum(p1_binary), len(p1_binary))
                p2_ci = self.wilson_confidence_interval(np.sum(p2_binary), len(p2_binary))
                
                analysis[error_type] = {
                    'pipeline1_rate': p1_rate,
                    'pipeline1_ci': p1_ci,
                    'pipeline2_rate': p2_rate,
                    'pipeline2_ci': p2_ci,
                    'mcnemar_test': mcnemar_result,
                    'total_p1_errors': np.sum(p1_errors),
                    'total_p2_errors': np.sum(p2_errors)
                }
        
        return analysis
    
    def multiple_comparison_correction(self, p_values: List[float], method: str = 'fdr_bh') -> Dict:
        """
        Apply multiple comparison correction using Benjamini-Hochberg FDR
        """
        rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            p_values, alpha=0.05, method=method
        )
        
        return {
            'original_p_values': p_values,
            'corrected_p_values': p_corrected.tolist(),
            'rejected_null': rejected.tolist(),
            'method': method,
            'alpha_level': 0.05
        }
    
    def generate_comprehensive_report(self, results_df: pd.DataFrame) -> Dict:
        """Generate comprehensive statistical analysis report"""
        
        print("Starting comprehensive statistical analysis...")
        
        # 1. Accuracy Analysis
        print("Analyzing accuracy metrics...")
        accuracy_analysis = self.analyze_accuracy_metrics(results_df)
        
        # 2. Latency Analysis
        print("Analyzing latency metrics...")
        latency_analysis = self.analyze_latency_metrics(results_df)
        
        # 3. GPU Analysis
        print("Analyzing GPU metrics...")
        gpu_analysis = self.analyze_gpu_metrics(results_df)
        
        # 4. Error Typology Analysis
        print("Analyzing error typology...")
        error_analysis = self.analyze_error_typology(results_df)
        
        # 5. Multiple Comparison Correction
        print("Applying multiple comparison correction...")
        
        # Collect all p-values for correction
        all_p_values = []
        p_value_labels = []
        
        # From accuracy analysis
        for complexity in accuracy_analysis:
            for metric in ['execution_accuracy', 'exact_match']:
                if metric in accuracy_analysis[complexity]:
                    p_val = accuracy_analysis[complexity][metric]['mcnemar_test']['p_value']
                    all_p_values.append(p_val)
                    p_value_labels.append(f'{complexity}_{metric}_mcnemar')
        
        # From latency analysis
        for complexity in latency_analysis:
            p_val = latency_analysis[complexity]['test_result']['p_value']
            all_p_values.append(p_val)
            p_value_labels.append(f'{complexity}_latency')
        
        # Apply correction
        correction_results = self.multiple_comparison_correction(all_p_values)
        
        # Compile final report
        final_report = {
            'metadata': {
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'random_seed': self.random_seed,
                'sample_size': len(results_df),
                'complexity_distribution': results_df['complexity'].value_counts().to_dict()
            },
            'accuracy_analysis': accuracy_analysis,
            'latency_analysis': latency_analysis,
            'gpu_analysis': gpu_analysis,
            'error_analysis': error_analysis,
            'multiple_comparison_correction': {
                'results': correction_results,
                'p_value_labels': p_value_labels
            }
        }
        
        print("Statistical analysis completed!")
        return final_report

def create_sample_data(n_queries: int = 150) -> pd.DataFrame:
    """
    Create sample data for testing the statistical framework
    Simulates 150 Vietnamese queries with realistic performance patterns
    """
    np.random.seed(42)
    
    # Stratified sampling: 50 queries per complexity level
    complexities = ['Simple'] * 50 + ['Medium'] * 50 + ['Complex'] * 50
    
    data = []
    
    for i, complexity in enumerate(complexities):
        # Simulate realistic performance differences
        if complexity == 'Simple':
            p1_ex_prob, p2_ex_prob = 0.85, 0.80
            p1_em_prob, p2_em_prob = 0.60, 0.50
            p1_latency_mean, p2_latency_mean = 600, 900
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
        
        # Generate latency (average of 3 replicates)
        p1_latencies = np.random.normal(p1_latency_mean, p1_latency_mean * 0.2, 3)
        p2_latencies = np.random.normal(p2_latency_mean, p2_latency_mean * 0.2, 3)
        
        p1_latency = np.mean(p1_latencies)
        p2_latency = np.mean(p2_latencies)
        
        # Generate GPU metrics
        p1_gpu = np.random.normal(4.2, 0.5)
        p2_gpu = np.random.normal(6.8, 0.8)
        
        # Generate error counts
        p1_tonal = np.random.poisson(0.1 if complexity == 'Simple' else 0.3)
        p2_tonal = np.random.poisson(0.15 if complexity == 'Simple' else 0.4)
        
        data.append({
            'query_id': f'Q{i+1:03d}',
            'complexity': complexity,
            'vietnamese_query': f'Sample Vietnamese query {i+1}',
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

if __name__ == "__main__":
    # Example usage
    print("Vietnamese NL2SQL Statistical Analysis Framework")
    print("=" * 50)
    
    # Create sample data
    sample_data = create_sample_data(150)
    print(f"Generated sample data: {len(sample_data)} queries")
    
    # Initialize statistical analyzer
    analyzer = VietnameseNL2SQLStatistics(random_seed=42)
    
    # Run comprehensive analysis
    report = analyzer.generate_comprehensive_report(sample_data)
    
    # Display key results
    print("\nKey Statistical Results:")
    print("-" * 30)
    
    # Overall accuracy results
    overall_ex = report['accuracy_analysis']['Overall']['execution_accuracy']
    print(f"Overall Execution Accuracy:")
    print(f"  {overall_ex['report_template']}")
    
    # Overall latency results
    overall_latency = report['latency_analysis']['Overall']['test_result']
    print(f"\nOverall Latency Analysis:")
    print(f"  Test: {overall_latency['test_type']}")
    print(f"  p-value: {overall_latency['p_value']:.3f}")
    print(f"  Effect size ({overall_latency['effect_size_name']}): {overall_latency['effect_size']:.3f}")
