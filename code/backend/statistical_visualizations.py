"""
Statistical Visualization Charts for Vietnamese NL2SQL Pipeline Evaluation
Generates publication-ready charts with proper statistical annotations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.patches as mpatches
from scipy import stats

# Set publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class VietnameseNL2SQLVisualizations:
    """Generate statistical visualizations for Vietnamese NL2SQL analysis"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'pipeline1': '#2E86AB',  # Blue
            'pipeline2': '#A23B72',  # Purple
            'simple': '#F18F01',     # Orange
            'medium': '#C73E1D',     # Red
            'complex': '#592E83'     # Dark Purple
        }
    
    def plot_accuracy_vs_complexity(self, accuracy_analysis: Dict, save_path: str = None) -> plt.Figure:
        """
        Generate accuracy vs query complexity chart with error bars
        Shows EX and EM metrics across Simple/Medium/Complex queries
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        complexities = ['Simple', 'Medium', 'Complex']
        
        # Execution Accuracy (EX) Chart
        p1_ex_rates = []
        p1_ex_cis = []
        p2_ex_rates = []
        p2_ex_cis = []
        
        for complexity in complexities:
            if complexity in accuracy_analysis:
                ex_data = accuracy_analysis[complexity]['execution_accuracy']
                p1_ex_rates.append(ex_data['pipeline1_rate'] * 100)
                p1_ex_cis.append([
                    (ex_data['pipeline1_rate'] - ex_data['pipeline1_ci'][0]) * 100,
                    (ex_data['pipeline1_ci'][1] - ex_data['pipeline1_rate']) * 100
                ])
                p2_ex_rates.append(ex_data['pipeline2_rate'] * 100)
                p2_ex_cis.append([
                    (ex_data['pipeline2_rate'] - ex_data['pipeline2_ci'][0]) * 100,
                    (ex_data['pipeline2_ci'][1] - ex_data['pipeline2_rate']) * 100
                ])
        
        x = np.arange(len(complexities))
        width = 0.35
        
        # Plot EX bars with error bars
        p1_ex_cis = np.array(p1_ex_cis).T
        p2_ex_cis = np.array(p2_ex_cis).T
        
        bars1 = ax1.bar(x - width/2, p1_ex_rates, width, 
                       yerr=p1_ex_cis, capsize=5,
                       label='Pipeline 1 (End-to-End Vietnamese)', 
                       color=self.colors['pipeline1'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, p2_ex_rates, width,
                       yerr=p2_ex_cis, capsize=5,
                       label='Pipeline 2 (Hybrid Translation)', 
                       color=self.colors['pipeline2'], alpha=0.8)
        
        ax1.set_xlabel('Query Complexity', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Execution Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Execution Accuracy vs Query Complexity\n(with 95% Wilson Confidence Intervals)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(complexities)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + 2,
                    f'{height1:.1f}%', ha='center', va='bottom', fontweight='bold')
            ax1.text(bar2.get_x() + bar2.get_width()/2., height2 + 2,
                    f'{height2:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Exact Match (EM) Chart
        p1_em_rates = []
        p1_em_cis = []
        p2_em_rates = []
        p2_em_cis = []
        
        for complexity in complexities:
            if complexity in accuracy_analysis:
                em_data = accuracy_analysis[complexity]['exact_match']
                p1_em_rates.append(em_data['pipeline1_rate'] * 100)
                p1_em_cis.append([
                    (em_data['pipeline1_rate'] - em_data['pipeline1_ci'][0]) * 100,
                    (em_data['pipeline1_ci'][1] - em_data['pipeline1_rate']) * 100
                ])
                p2_em_rates.append(em_data['pipeline2_rate'] * 100)
                p2_em_cis.append([
                    (em_data['pipeline2_rate'] - em_data['pipeline2_ci'][0]) * 100,
                    (em_data['pipeline2_ci'][1] - em_data['pipeline2_rate']) * 100
                ])
        
        p1_em_cis = np.array(p1_em_cis).T
        p2_em_cis = np.array(p2_em_cis).T
        
        bars3 = ax2.bar(x - width/2, p1_em_rates, width,
                       yerr=p1_em_cis, capsize=5,
                       label='Pipeline 1 (End-to-End Vietnamese)', 
                       color=self.colors['pipeline1'], alpha=0.8)
        bars4 = ax2.bar(x + width/2, p2_em_rates, width,
                       yerr=p2_em_cis, capsize=5,
                       label='Pipeline 2 (Hybrid Translation)', 
                       color=self.colors['pipeline2'], alpha=0.8)
        
        ax2.set_xlabel('Query Complexity', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Exact Match (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Exact Match vs Query Complexity\n(with 95% Wilson Confidence Intervals)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(complexities)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Add value labels on bars
        for i, (bar3, bar4) in enumerate(zip(bars3, bars4)):
            height3 = bar3.get_height()
            height4 = bar4.get_height()
            ax2.text(bar3.get_x() + bar3.get_width()/2., height3 + 1,
                    f'{height3:.1f}%', ha='center', va='bottom', fontweight='bold')
            ax2.text(bar4.get_x() + bar4.get_width()/2., height4 + 1,
                    f'{height4:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_error_type_breakdown(self, error_analysis: Dict, save_path: str = None) -> plt.Figure:
        """
        Generate error type breakdown chart as stacked 100% bars
        Shows distribution of different error types by pipeline
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract error data
        error_types = ['tonal_accent_errors', 'compound_word_errors', 
                      'sql_syntax_errors', 'schema_logic_errors']
        error_labels = ['Tonal/Accent\nErrors', 'Compound Word\nErrors', 
                       'SQL Syntax\nErrors', 'Schema Logic\nErrors']
        
        p1_totals = []
        p2_totals = []
        
        for error_type in error_types:
            if error_type in error_analysis:
                p1_totals.append(error_analysis[error_type]['total_p1_errors'])
                p2_totals.append(error_analysis[error_type]['total_p2_errors'])
            else:
                p1_totals.append(0)
                p2_totals.append(0)
        
        # Calculate percentages
        p1_sum = sum(p1_totals)
        p2_sum = sum(p2_totals)
        
        if p1_sum > 0:
            p1_percentages = [x/p1_sum * 100 for x in p1_totals]
        else:
            p1_percentages = [0] * len(p1_totals)
            
        if p2_sum > 0:
            p2_percentages = [x/p2_sum * 100 for x in p2_totals]
        else:
            p2_percentages = [0] * len(p2_totals)
        
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
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_latency_violin(self, latency_analysis: Dict, results_df: pd.DataFrame, save_path: str = None) -> plt.Figure:
        """
        Generate paired violin plot for latency analysis
        Shows distribution of latency by pipeline and complexity
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data for violin plot
        data_for_plot = []
        
        complexities = ['Simple', 'Medium', 'Complex']
        
        for complexity in complexities:
            subset = results_df[results_df['complexity'] == complexity]
            
            for _, row in subset.iterrows():
                data_for_plot.append({
                    'Complexity': complexity,
                    'Pipeline': 'Pipeline 1',
                    'Latency (ms)': row['pipeline1_latency_ms']
                })
                data_for_plot.append({
                    'Complexity': complexity,
                    'Pipeline': 'Pipeline 2', 
                    'Latency (ms)': row['pipeline2_latency_ms']
                })
        
        plot_df = pd.DataFrame(data_for_plot)
        
        # Create violin plot
        sns.violinplot(data=plot_df, x='Complexity', y='Latency (ms)', 
                      hue='Pipeline', split=False, ax=ax, palette=[self.colors['pipeline1'], self.colors['pipeline2']])
        
        # Add median lines and statistical annotations
        for i, complexity in enumerate(complexities):
            if complexity in latency_analysis:
                stats_data = latency_analysis[complexity]['descriptive_stats']
                test_result = latency_analysis[complexity]['test_result']
                
                # Add median markers
                p1_median = stats_data['pipeline1_median']
                p2_median = stats_data['pipeline2_median']
                
                ax.plot(i-0.2, p1_median, 'o', color='white', markersize=8, markeredgecolor='black', markeredgewidth=2)
                ax.plot(i+0.2, p2_median, 'o', color='white', markersize=8, markeredgecolor='black', markeredgewidth=2)
                
                # Add p-value annotation
                max_y = max(plot_df[plot_df['Complexity'] == complexity]['Latency (ms)'])
                p_val = test_result['p_value']
                significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                
                ax.text(i, max_y * 1.1, f'p={p_val:.3f}\n({significance})', 
                       ha='center', va='bottom', fontweight='bold', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax.set_xlabel('Query Complexity', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Latency Distribution by Pipeline and Complexity\n(Violin Plot with Median Markers)', 
                    fontsize=14, fontweight='bold')
        ax.legend(title='Pipeline', fontsize=11, title_fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_bland_altman(self, results_df: pd.DataFrame, save_path: str = None) -> plt.Figure:
        """
        Generate Bland-Altman plot for latency differences
        Shows agreement between pipeline latencies
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate differences and means
        p1_latency = results_df['pipeline1_latency_ms'].values
        p2_latency = results_df['pipeline2_latency_ms'].values
        
        differences = p1_latency - p2_latency
        means = (p1_latency + p2_latency) / 2
        
        # Calculate statistics
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        # Plot data points colored by complexity
        complexity_colors = {'Simple': self.colors['simple'], 
                           'Medium': self.colors['medium'], 
                           'Complex': self.colors['complex']}
        
        for complexity in ['Simple', 'Medium', 'Complex']:
            mask = results_df['complexity'] == complexity
            ax.scatter(means[mask], differences[mask], 
                      c=complexity_colors[complexity], label=complexity, 
                      alpha=0.7, s=50)
        
        # Add reference lines
        ax.axhline(mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean difference: {mean_diff:.1f} ms')
        ax.axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', linewidth=1, 
                  label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.1f} ms')
        ax.axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--', linewidth=1,
                  label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.1f} ms')
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Mean Latency: (Pipeline1 + Pipeline2) / 2 (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Difference: Pipeline1 - Pipeline2 (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Bland-Altman Plot: Pipeline Latency Agreement\n(95% Limits of Agreement)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def generate_all_charts(self, statistical_report: Dict, results_df: pd.DataFrame, output_dir: str = "charts/") -> Dict[str, str]:
        """
        Generate all statistical visualization charts
        Returns dictionary of chart names and file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        chart_paths = {}
        
        # 1. Accuracy vs Complexity Chart
        print("Generating accuracy vs complexity chart...")
        fig1 = self.plot_accuracy_vs_complexity(
            statistical_report['accuracy_analysis'], 
            save_path=f"{output_dir}/accuracy_vs_complexity.png"
        )
        chart_paths['accuracy_vs_complexity'] = f"{output_dir}/accuracy_vs_complexity.png"
        plt.close(fig1)
        
        # 2. Error Type Breakdown Chart
        print("Generating error type breakdown chart...")
        fig2 = self.plot_error_type_breakdown(
            statistical_report['error_analysis'],
            save_path=f"{output_dir}/error_type_breakdown.png"
        )
        chart_paths['error_type_breakdown'] = f"{output_dir}/error_type_breakdown.png"
        plt.close(fig2)
        
        # 3. Latency Violin Plot
        print("Generating latency violin plot...")
        fig3 = self.plot_latency_violin(
            statistical_report['latency_analysis'],
            results_df,
            save_path=f"{output_dir}/latency_violin_plot.png"
        )
        chart_paths['latency_violin'] = f"{output_dir}/latency_violin_plot.png"
        plt.close(fig3)
        
        # 4. Bland-Altman Plot
        print("Generating Bland-Altman plot...")
        fig4 = self.plot_bland_altman(
            results_df,
            save_path=f"{output_dir}/bland_altman_plot.png"
        )
        chart_paths['bland_altman'] = f"{output_dir}/bland_altman_plot.png"
        plt.close(fig4)
        
        print(f"All charts generated and saved to {output_dir}")
        return chart_paths

def create_statistical_charts_demo():
    """Demo function to generate charts with sample data"""
    from statistical_analysis import VietnameseNL2SQLStatistics, create_sample_data
    
    # Generate sample data
    sample_data = create_sample_data(150)
    
    # Run statistical analysis
    analyzer = VietnameseNL2SQLStatistics(random_seed=42)
    report = analyzer.generate_comprehensive_report(sample_data)
    
    # Generate visualizations
    visualizer = VietnameseNL2SQLVisualizations()
    chart_paths = visualizer.generate_all_charts(report, sample_data, "statistical_charts/")
    
    return report, chart_paths

if __name__ == "__main__":
    print("Vietnamese NL2SQL Statistical Visualizations")
    print("=" * 50)
    
    # Run demo
    report, chart_paths = create_statistical_charts_demo()
    
    print("\nGenerated Charts:")
    for chart_name, path in chart_paths.items():
        print(f"  {chart_name}: {path}")
    
    print("\nCharts generated successfully!")
