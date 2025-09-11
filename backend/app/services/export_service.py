"""
Export service for Interactive Spectral Clustering Platform.

Provides comprehensive export functionality including CSV exports,
HTML reports, and bundled ZIP downloads with results and metadata.
"""

import os
import json
import zipfile
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from jinja2 import Template
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import asdict
import logging

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie

logger = logging.getLogger(__name__)

class ExportService:
    """Service for exporting clustering results and analysis reports."""
    
    def __init__(self):
        """Initialize the export service."""
        self.html_template = self._get_html_template()
        
        # PDF styles
        self.pdf_styles = getSampleStyleSheet()
        self.pdf_styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.pdf_styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2c3e50'),
            alignment=TA_CENTER
        ))
        self.pdf_styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.pdf_styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            spaceBefore=20,
            textColor=colors.HexColor('#34495e'),
            borderWidth=1,
            borderColor=colors.HexColor('#3498db'),
            borderPadding=5
        ))
        self.pdf_styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.pdf_styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            spaceBefore=15,
            textColor=colors.HexColor('#5a6c7d')
        ))
        self.pdf_styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.pdf_styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            alignment=TA_JUSTIFY
        ))
    
    def export_labels_csv(
        self, 
        labels: np.ndarray, 
        original_data: Optional[np.ndarray] = None,
        column_names: Optional[List[str]] = None
    ) -> str:
        """
        Export clustering labels as CSV format.
        
        Args:
            labels: Cluster labels array
            original_data: Optional original dataset
            column_names: Optional column names for original data
            
        Returns:
            CSV content as string
        """
        try:
            # Create DataFrame with labels
            df_data = {"cluster_label": labels}
            
            # Add original data if provided
            if original_data is not None:
                if column_names is None:
                    column_names = [f"feature_{i}" for i in range(original_data.shape[1])]
                
                for i, col_name in enumerate(column_names):
                    df_data[col_name] = original_data[:, i]
            
            # Add sample index
            df_data["sample_index"] = range(len(labels))
            
            df = pd.DataFrame(df_data)
            
            # Convert to CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            return csv_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error exporting labels to CSV: {str(e)}")
            raise ValueError(f"Failed to export labels: {str(e)}")
    
    def generate_clustering_report(
        self,
        run_data: Dict[str, Any],
        labels: np.ndarray,
        dataset_stats: Optional[Dict[str, Any]] = None,
        preprocessing_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate comprehensive HTML clustering report.
        
        Args:
            run_data: Clustering run information
            labels: Cluster labels
            dataset_stats: Dataset statistics
            preprocessing_info: Preprocessing information
            
        Returns:
            HTML report as string
        """
        try:
            # Prepare report data
            report_data = self._prepare_report_data(
                run_data, labels, dataset_stats, preprocessing_info
            )
            
            # Generate visualizations
            visualizations = self._generate_report_visualizations(labels, run_data)
            report_data.update(visualizations)
            
            # Render HTML template
            template = Template(self.html_template)
            html_content = template.render(**report_data)
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error generating clustering report: {str(e)}")
            raise ValueError(f"Failed to generate report: {str(e)}")
    
    def create_export_bundle(
        self,
        run_data: Dict[str, Any],
        labels: np.ndarray,
        original_data: Optional[np.ndarray] = None,
        dataset_stats: Optional[Dict[str, Any]] = None,
        preprocessing_info: Optional[Dict[str, Any]] = None,
        column_names: Optional[List[str]] = None
    ) -> bytes:
        """
        Create ZIP bundle with all export materials.
        
        Args:
            run_data: Clustering run information
            labels: Cluster labels
            original_data: Original dataset
            dataset_stats: Dataset statistics
            preprocessing_info: Preprocessing information
            column_names: Column names for data
            
        Returns:
            ZIP file content as bytes
        """
        try:
            # Create temporary directory for files
            with tempfile.TemporaryDirectory() as temp_dir:
                
                # 1. Export labels CSV
                labels_csv = self.export_labels_csv(labels, original_data, column_names)
                labels_path = os.path.join(temp_dir, "cluster_labels.csv")
                with open(labels_path, 'w', encoding='utf-8') as f:
                    f.write(labels_csv)
                
                # 2. Generate HTML report
                html_report = self.generate_clustering_report(
                    run_data, labels, dataset_stats, preprocessing_info
                )
                report_path = os.path.join(temp_dir, "clustering_report.html")
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(html_report)
                
                # 3. Export metadata JSON
                metadata = {
                    "run_info": run_data,
                    "dataset_stats": dataset_stats,
                    "preprocessing_info": preprocessing_info,
                    "export_timestamp": datetime.now().isoformat(),
                    "cluster_summary": self._get_cluster_summary(labels)
                }
                metadata_path = os.path.join(temp_dir, "metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                # 4. Create README
                readme_content = self._generate_readme(run_data, labels)
                readme_path = os.path.join(temp_dir, "README.txt")
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(readme_content)
                
                # 5. Create ZIP bundle
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    zip_file.write(labels_path, "cluster_labels.csv")
                    zip_file.write(report_path, "clustering_report.html")
                    zip_file.write(metadata_path, "metadata.json")
                    zip_file.write(readme_path, "README.txt")
                
                return zip_buffer.getvalue()
                
        except Exception as e:
            logger.error(f"Error creating export bundle: {str(e)}")
            raise ValueError(f"Failed to create export bundle: {str(e)}")
    
    def _prepare_report_data(
        self,
        run_data: Dict[str, Any],
        labels: np.ndarray,
        dataset_stats: Optional[Dict[str, Any]] = None,
        preprocessing_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare data for HTML report template."""
        
        # Basic run information
        report_data = {
            "title": f"Clustering Report - Run {run_data.get('id', 'Unknown')}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": run_data.get("algorithm", "Unknown"),
            "parameters": run_data.get("parameters", {}),
            "dataset_name": run_data.get("dataset_name", "Unknown Dataset")
        }
        
        # Cluster analysis
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
        n_noise = np.sum(labels == -1) if -1 in labels else 0
        
        cluster_summary = {
            "n_clusters": n_clusters,
            "n_samples": len(labels),
            "n_noise": n_noise,
            "noise_percentage": (n_noise / len(labels) * 100) if len(labels) > 0 else 0
        }
        
        # Cluster sizes
        cluster_sizes = {}
        for label in unique_labels:
            if label >= 0:  # Exclude noise
                size = np.sum(labels == label)
                cluster_sizes[f"Cluster {label}"] = size
        
        report_data.update({
            "cluster_summary": cluster_summary,
            "cluster_sizes": cluster_sizes,
            "dataset_stats": dataset_stats or {},
            "preprocessing_info": preprocessing_info or {}
        })
        
        return report_data
    
    def _generate_report_visualizations(
        self, 
        labels: np.ndarray, 
        run_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate base64-encoded visualizations for the report."""
        visualizations = {}
        
        try:
            # Cluster size distribution
            unique_labels = np.unique(labels)
            cluster_labels = unique_labels[unique_labels >= 0]
            
            if len(cluster_labels) > 0:
                cluster_sizes = [np.sum(labels == label) for label in cluster_labels]
                
                # Bar chart of cluster sizes
                plt.figure(figsize=(10, 6))
                plt.bar([f"Cluster {i}" for i in cluster_labels], cluster_sizes)
                plt.title("Cluster Size Distribution")
                plt.xlabel("Cluster")
                plt.ylabel("Number of Samples")
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Convert to base64
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                visualizations["cluster_sizes_chart"] = img_base64
                plt.close()
                
                # Pie chart for cluster proportions
                if len(cluster_sizes) <= 10:  # Only for reasonable number of clusters
                    plt.figure(figsize=(8, 8))
                    labels_pie = [f"Cluster {i}" for i in cluster_labels]
                    plt.pie(cluster_sizes, labels=labels_pie, autopct='%1.1f%%', startangle=90)
                    plt.title("Cluster Proportion")
                    plt.axis('equal')
                    
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                    img_buffer.seek(0)
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    visualizations["cluster_proportions_chart"] = img_base64
                    plt.close()
            
        except Exception as e:
            logger.warning(f"Error generating visualizations: {str(e)}")
            visualizations["error"] = f"Could not generate visualizations: {str(e)}"
        
        return visualizations
    
    def _get_cluster_summary(self, labels: np.ndarray) -> Dict[str, Any]:
        """Get summary statistics for clusters."""
        unique_labels = np.unique(labels)
        cluster_labels = unique_labels[unique_labels >= 0]
        
        summary = {
            "total_clusters": len(cluster_labels),
            "total_samples": len(labels),
            "noise_samples": np.sum(labels == -1) if -1 in labels else 0,
            "cluster_sizes": {}
        }
        
        for label in cluster_labels:
            size = np.sum(labels == label)
            summary["cluster_sizes"][int(label)] = int(size)
        
        return summary

    def generate_pdf_report(
        self,
        report_type: str,
        run_data: Dict[str, Any],
        labels: Optional[np.ndarray] = None,
        dataset_stats: Optional[Dict[str, Any]] = None,
        preprocessing_info: Optional[Dict[str, Any]] = None,
        experiment_results: Optional[List[Dict[str, Any]]] = None
    ) -> bytes:
        """
        Generate PDF report based on report type.
        
        Args:
            report_type: Type of report ('executive', 'detailed', 'technical', 'comparison')
            run_data: Clustering run information
            labels: Cluster labels
            dataset_stats: Dataset statistics
            preprocessing_info: Preprocessing information
            experiment_results: List of experiment results for comparison
            
        Returns:
            PDF content as bytes
        """
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            story = []
            
            # Add content based on report type
            if report_type == 'executive':
                story.extend(self._build_executive_summary(run_data, labels, dataset_stats))
            elif report_type == 'detailed':
                story.extend(self._build_detailed_analysis(run_data, labels, dataset_stats, preprocessing_info))
            elif report_type == 'technical':
                story.extend(self._build_technical_report(run_data, labels, dataset_stats, preprocessing_info))
            elif report_type == 'comparison':
                story.extend(self._build_algorithm_comparison(experiment_results or [], dataset_stats))
            else:
                raise ValueError(f"Unknown report type: {report_type}")
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            raise ValueError(f"Failed to generate PDF report: {str(e)}")

    def _build_executive_summary(
        self, 
        run_data: Dict[str, Any], 
        labels: Optional[np.ndarray], 
        dataset_stats: Optional[Dict[str, Any]]
    ) -> List:
        """Build executive summary sections for PDF."""
        story = []
        
        # Title page
        story.append(Paragraph("Executive Summary", self.pdf_styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Overview
        story.append(Paragraph("Project Overview", self.pdf_styles['CustomHeading1']))
        story.append(Paragraph(
            f"This report presents the results of clustering analysis performed on the dataset "
            f"'{run_data.get('dataset_name', 'Unknown Dataset')}' using the "
            f"{run_data.get('algorithm', 'Unknown').upper()} algorithm.",
            self.pdf_styles['CustomBody']
        ))
        
        # Key findings
        if labels is not None:
            cluster_summary = self._get_cluster_summary(labels)
            story.append(Paragraph("Key Findings", self.pdf_styles['CustomHeading2']))
            story.append(Paragraph(
                f"• {cluster_summary['total_clusters']} distinct clusters were identified<br/>"
                f"• {cluster_summary['total_samples']} data points were analyzed<br/>"
                f"• {cluster_summary.get('noise_samples', 0)} points classified as noise<br/>"
                f"• Analysis completed on {datetime.now().strftime('%B %d, %Y')}",
                self.pdf_styles['CustomBody']
            ))
        
        # Dataset summary
        if dataset_stats:
            story.append(Paragraph("Dataset Summary", self.pdf_styles['CustomHeading2']))
            story.append(Paragraph(
                f"• Records: {dataset_stats.get('shape', [0, 0])[0]:,}<br/>"
                f"• Features: {dataset_stats.get('shape', [0, 0])[1]}<br/>"
                f"• Size: {dataset_stats.get('memory_usage', 0):.2f} MB",
                self.pdf_styles['CustomBody']
            ))
        
        # Recommendations
        story.append(Paragraph("Recommendations", self.pdf_styles['CustomHeading2']))
        story.append(Paragraph(
            "Based on the clustering analysis, we recommend further investigation of the identified "
            "patterns for business insights and potential actions. Detailed technical analysis "
            "is available in the comprehensive technical report.",
            self.pdf_styles['CustomBody']
        ))
        
        return story

    def _build_detailed_analysis(
        self, 
        run_data: Dict[str, Any], 
        labels: Optional[np.ndarray], 
        dataset_stats: Optional[Dict[str, Any]], 
        preprocessing_info: Optional[Dict[str, Any]]
    ) -> List:
        """Build detailed analysis sections for PDF."""
        story = []
        
        # Title
        story.append(Paragraph("Detailed Clustering Analysis", self.pdf_styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Executive Summary
        story.extend(self._build_executive_summary(run_data, labels, dataset_stats))
        story.append(PageBreak())
        
        # Methodology
        story.append(Paragraph("Methodology", self.pdf_styles['CustomHeading1']))
        story.append(Paragraph(
            f"Algorithm: {run_data.get('algorithm', 'Unknown').upper()}<br/>"
            f"Implementation: Python scikit-learn library<br/>"
            f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.pdf_styles['CustomBody']
        ))
        
        # Parameters
        if run_data.get('parameters'):
            story.append(Paragraph("Algorithm Parameters", self.pdf_styles['CustomHeading2']))
            params_data = []
            for key, value in run_data['parameters'].items():
                params_data.append([key, str(value)])
            
            if params_data:
                table = Table(params_data, colWidths=[2*inch, 3*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
        
        # Results
        if labels is not None:
            story.append(Paragraph("Clustering Results", self.pdf_styles['CustomHeading1']))
            cluster_summary = self._get_cluster_summary(labels)
            
            results_data = [
                ['Metric', 'Value'],
                ['Total Clusters', str(cluster_summary['total_clusters'])],
                ['Total Samples', str(cluster_summary['total_samples'])],
                ['Noise Points', str(cluster_summary.get('noise_samples', 0))]
            ]
            
            table = Table(results_data, colWidths=[2.5*inch, 2.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        
        return story

    def _build_technical_report(
        self, 
        run_data: Dict[str, Any], 
        labels: Optional[np.ndarray], 
        dataset_stats: Optional[Dict[str, Any]], 
        preprocessing_info: Optional[Dict[str, Any]]
    ) -> List:
        """Build technical report sections for PDF."""
        story = []
        
        # Title
        story.append(Paragraph("Technical Clustering Report", self.pdf_styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Include detailed analysis
        story.extend(self._build_detailed_analysis(run_data, labels, dataset_stats, preprocessing_info))
        story.append(PageBreak())
        
        # Technical Implementation Details
        story.append(Paragraph("Technical Implementation", self.pdf_styles['CustomHeading1']))
        
        # Preprocessing details
        if preprocessing_info:
            story.append(Paragraph("Data Preprocessing", self.pdf_styles['CustomHeading2']))
            
            if preprocessing_info.get('steps_applied'):
                story.append(Paragraph("Applied Preprocessing Steps:", self.pdf_styles['CustomBody']))
                for step in preprocessing_info['steps_applied']:
                    story.append(Paragraph(f"• {step}", self.pdf_styles['CustomBody']))
            
            if preprocessing_info.get('removed_columns'):
                story.append(Paragraph(
                    f"Removed Columns: {len(preprocessing_info['removed_columns'])} columns were "
                    f"removed during preprocessing due to missing data or low variance.",
                    self.pdf_styles['CustomBody']
                ))
        
        # Performance Metrics
        story.append(Paragraph("Performance Metrics", self.pdf_styles['CustomHeading2']))
        story.append(Paragraph(
            "Detailed performance metrics including silhouette score, Davies-Bouldin index, "
            "and Calinski-Harabasz index would be displayed here in a production environment.",
            self.pdf_styles['CustomBody']
        ))
        
        # Computational Details
        story.append(Paragraph("Computational Details", self.pdf_styles['CustomHeading2']))
        story.append(Paragraph(
            f"• Algorithm: {run_data.get('algorithm', 'Unknown')}<br/>"
            f"• Library: scikit-learn<br/>"
            f"• Python Version: 3.8+<br/>"
            f"• Execution Time: {run_data.get('execution_time', 'N/A')} seconds",
            self.pdf_styles['CustomBody']
        ))
        
        return story

    def _build_algorithm_comparison(
        self, 
        experiment_results: List[Dict[str, Any]], 
        dataset_stats: Optional[Dict[str, Any]]
    ) -> List:
        """Build algorithm comparison sections for PDF."""
        story = []
        
        # Title
        story.append(Paragraph("Algorithm Comparison Report", self.pdf_styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Overview
        story.append(Paragraph("Comparison Overview", self.pdf_styles['CustomHeading1']))
        story.append(Paragraph(
            f"This report compares the performance of {len(experiment_results)} different "
            f"clustering algorithms on the provided dataset.",
            self.pdf_styles['CustomBody']
        ))
        
        # Dataset info
        if dataset_stats:
            story.append(Paragraph("Dataset Information", self.pdf_styles['CustomHeading2']))
            story.append(Paragraph(
                f"• Records: {dataset_stats.get('shape', [0, 0])[0]:,}<br/>"
                f"• Features: {dataset_stats.get('shape', [0, 0])[1]}<br/>"
                f"• Size: {dataset_stats.get('memory_usage', 0):.2f} MB",
                self.pdf_styles['CustomBody']
            ))
        
        # Results comparison
        if experiment_results:
            story.append(Paragraph("Algorithm Performance Comparison", self.pdf_styles['CustomHeading1']))
            
            # Create comparison table
            table_data = [['Algorithm', 'Clusters', 'Silhouette Score', 'Execution Time']]
            
            for result in experiment_results[:10]:  # Limit to top 10
                table_data.append([
                    result.get('algorithm', 'Unknown'),
                    str(result.get('n_clusters', 'N/A')),
                    f"{result.get('silhouette_score', 0):.3f}",
                    f"{result.get('execution_time', 0):.2f}s"
                ])
            
            table = Table(table_data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
        
        # Recommendations
        story.append(Paragraph("Recommendations", self.pdf_styles['CustomHeading2']))
        story.append(Paragraph(
            "Based on the comparison analysis, recommendations for the most suitable algorithm "
            "would be provided here, considering factors such as clustering quality, "
            "computational efficiency, and interpretability.",
            self.pdf_styles['CustomBody']
        ))
        
        return story
    
    def _generate_readme(self, run_data: Dict[str, Any], labels: np.ndarray) -> str:
        """Generate README file for export bundle."""
        
        cluster_summary = self._get_cluster_summary(labels)
        
        readme = f"""
# Clustering Results Export Bundle

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Contents

1. **cluster_labels.csv** - Clustering results with original data
2. **clustering_report.html** - Comprehensive analysis report (print-ready)
3. **metadata.json** - Complete run metadata and statistics
4. **README.txt** - This file

## Run Information

- Algorithm: {run_data.get('algorithm', 'Unknown')}
- Dataset: {run_data.get('dataset_name', 'Unknown')}
- Parameters: {json.dumps(run_data.get('parameters', {}), indent=2)}

## Results Summary

- Total Samples: {cluster_summary['total_samples']}
- Clusters Found: {cluster_summary['total_clusters']}
- Noise Samples: {cluster_summary['noise_samples']}

## Usage

- Open `clustering_report.html` in any web browser for detailed analysis
- Import `cluster_labels.csv` into Excel, R, Python, or other analysis tools
- Use `metadata.json` for programmatic access to all run information

## Support

For questions about this export or the clustering platform, please refer to the
Interactive Spectral Clustering Platform documentation.
"""
        return readme.strip()
    
    def _get_html_template(self) -> str:
        """Get HTML template for clustering reports."""
        
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        @media print {
            .no-print { display: none; }
            body { font-size: 12pt; }
            h1 { page-break-before: auto; }
            .section { page-break-inside: avoid; }
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #333;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        
        h2 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        
        h3 {
            color: #5a6c7d;
            margin-top: 25px;
            margin-bottom: 15px;
        }
        
        .header-info {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        
        .header-info p {
            margin: 5px 0;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        
        .stat-card h4 {
            margin-top: 0;
            color: #495057;
            border-bottom: 1px solid #dee2e6;
            padding-bottom: 5px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            font-weight: 500;
            color: #6c757d;
        }
        
        .metric-value {
            font-weight: bold;
            color: #495057;
        }
        
        .visualization {
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .visualization img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .parameters-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        
        .parameters-table th,
        .parameters-table td {
            border: 1px solid #dee2e6;
            padding: 12px;
            text-align: left;
        }
        
        .parameters-table th {
            background-color: #e9ecef;
            font-weight: 600;
            color: #495057;
        }
        
        .parameters-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        .cluster-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }
        
        .cluster-item {
            background: #e3f2fd;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            border: 1px solid #bbdefb;
        }
        
        .cluster-item strong {
            color: #1976d2;
        }
        
        .section {
            margin-bottom: 40px;
        }
        
        .alert {
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
            border-left: 4px solid #f39c12;
            background-color: #fef9e7;
        }
        
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        
        <div class="header-info">
            <p><strong>Generated:</strong> {{ timestamp }}</p>
            <p><strong>Dataset:</strong> {{ dataset_name }}</p>
            <p><strong>Algorithm:</strong> {{ algorithm }}</p>
        </div>

        <div class="section">
            <h2>Clustering Results</h2>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <h4>Cluster Summary</h4>
                    <div class="metric">
                        <span class="metric-label">Clusters Found:</span>
                        <span class="metric-value">{{ cluster_summary.n_clusters }}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total Samples:</span>
                        <span class="metric-value">{{ cluster_summary.n_samples }}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Noise Points:</span>
                        <span class="metric-value">{{ cluster_summary.n_noise }}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Noise Percentage:</span>
                        <span class="metric-value">{{ "%.2f"|format(cluster_summary.noise_percentage) }}%</span>
                    </div>
                </div>
                
                {% if cluster_sizes %}
                <div class="stat-card">
                    <h4>Cluster Sizes</h4>
                    <div class="cluster-list">
                        {% for cluster, size in cluster_sizes.items() %}
                        <div class="cluster-item">
                            <strong>{{ cluster }}</strong><br>
                            {{ size }} samples
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
            </div>
            
            {% if cluster_sizes_chart %}
            <div class="visualization">
                <h3>Cluster Size Distribution</h3>
                <img src="data:image/png;base64,{{ cluster_sizes_chart }}" alt="Cluster Size Distribution">
            </div>
            {% endif %}
            
            {% if cluster_proportions_chart %}
            <div class="visualization">
                <h3>Cluster Proportions</h3>
                <img src="data:image/png;base64,{{ cluster_proportions_chart }}" alt="Cluster Proportions">
            </div>
            {% endif %}
        </div>

        <div class="section">
            <h2>Algorithm Parameters</h2>
            <table class="parameters-table">
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    {% for param, value in parameters.items() %}
                    <tr>
                        <td>{{ param }}</td>
                        <td>{{ value }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        {% if dataset_stats %}
        <div class="section">
            <h2>Dataset Information</h2>
            
            {% if dataset_stats.shape %}
            <div class="stats-grid">
                <div class="stat-card">
                    <h4>Dataset Shape</h4>
                    <div class="metric">
                        <span class="metric-label">Rows:</span>
                        <span class="metric-value">{{ dataset_stats.shape[0] }}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Columns:</span>
                        <span class="metric-value">{{ dataset_stats.shape[1] }}</span>
                    </div>
                </div>
                
                {% if dataset_stats.memory_usage %}
                <div class="stat-card">
                    <h4>Memory Usage</h4>
                    <div class="metric">
                        <span class="metric-label">Size:</span>
                        <span class="metric-value">{{ "%.2f"|format(dataset_stats.memory_usage) }} MB</span>
                    </div>
                </div>
                {% endif %}
            </div>
            {% endif %}
        </div>
        {% endif %}

        {% if preprocessing_info %}
        <div class="section">
            <h2>Preprocessing Information</h2>
            
            {% if preprocessing_info.steps_applied %}
            <h3>Applied Steps:</h3>
            <ul>
                {% for step in preprocessing_info.steps_applied %}
                <li>{{ step }}</li>
                {% endfor %}
            </ul>
            {% endif %}
            
            {% if preprocessing_info.removed_columns %}
            <div class="alert">
                <strong>Note:</strong> {{ preprocessing_info.removed_columns|length }} columns were removed during preprocessing.
            </div>
            {% endif %}
        </div>
        {% endif %}

        <div class="footer">
            <p>Generated by Interactive Spectral Clustering Platform</p>
            <p>Report generated on {{ timestamp }}</p>
        </div>
    </div>
</body>
</html>
"""
