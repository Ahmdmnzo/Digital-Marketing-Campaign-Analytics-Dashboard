"""
Digital Marketing Campaign Analysis Script
==========================================
This script analyzes digital marketing campaign data to provide insights on:
- Campaign channel distribution and performance
- Customer segmentation using RFM analysis
- Customer Lifetime Value (CLV) calculations
- Cost Per Acquisition (CPA) and Return on Ad Spend (ROAS)
- Campaign effectiveness metrics
- Engagement correlation analysis
- Demographic-based channel recommendations
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class MarketingAnalyzer:
    """
    A comprehensive marketing campaign analysis tool
    """
    
    def __init__(self, data_path):
        """
        Initialize the analyzer with data
        
        Args:
            data_path (str): Path to the CSV file containing marketing data
        """
        self.data = pd.read_csv(data_path)
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare and clean the data for analysis"""
        # Add derived columns
        self.data['Gender_Numeric'] = self.data['Gender'].map({'Male': 1, 'Female': 0})
        self.data['Age_Groups'] = pd.cut(
            self.data['Age'], 
            bins=[0, 30, 45, 60, 100], 
            labels=['Young', 'Adult', 'Middle', 'Senior']
        )
        
        # RFM Analysis preparation
        self.data['Recency'] = self.data['LoyaltyPoints'].max() - self.data['LoyaltyPoints'] + 1
        self.data['Monetary'] = self.data['Income']
        self.data['Frequency'] = self.data['PreviousPurchases']
        
        # RFM Scores
        self.data['R_Score'] = pd.qcut(self.data['Recency'], 3, labels=['High', 'Medium', 'Low'])
        self.data['F_Score'] = pd.qcut(self.data['Frequency'], 3, labels=['High', 'Medium', 'Low'])
        self.data['M_Score'] = pd.qcut(self.data['Monetary'], 3, labels=['High', 'Medium', 'Low'])
        
        # RFM Segment
        self.data['RFM_Segment'] = (
            self.data['R_Score'].astype(str) + '_' + 
            self.data['F_Score'].astype(str) + '_' + 
            self.data['M_Score'].astype(str)
        )
        
        # CLV Calculations
        self.data['AOV_Proxy'] = (self.data['Income'] / 12) * 0.15
        self.data['Lifespan_Proxy'] = self.data['LoyaltyPoints'] / 1000
        self.data['CLV'] = self.data['AOV_Proxy'] * self.data['PreviousPurchases'] * self.data['Lifespan_Proxy']
        self.data['CLV_Basic'] = (self.data['Income'] / 12) * self.data['PreviousPurchases'] * 2
        self.data['CLV_Loyalty'] = self.data['Income'] * 0.1 * (self.data['LoyaltyPoints'] / 1000)
    
    def get_data_overview(self):
        """Get basic overview of the dataset"""
        print("=" * 50)
        print("DATA OVERVIEW")
        print("=" * 50)
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        print("\nData types and null values:")
        print(self.data.info())
        print("\nFirst 5 rows:")
        print(self.data.head())
    
    def analyze_campaign_channels(self):
        """Analyze campaign channel distribution"""
        print("\n" + "=" * 50)
        print("CAMPAIGN CHANNEL ANALYSIS")
        print("=" * 50)
        
        channel_counts = self.data.groupby('CampaignChannel')['CustomerID'].count()
        channel_percentages = (channel_counts / self.data['CustomerID'].count() * 100).round(2)
        
        print("Channel Distribution:")
        for channel, count in channel_counts.items():
            percentage = channel_percentages[channel]
            print(f"  {channel}: {count} customers ({percentage}%)")
        
        return channel_counts, channel_percentages
    
    def calculate_key_metrics(self):
        """Calculate key marketing metrics"""
        print("\n" + "=" * 50)
        print("KEY METRICS")
        print("=" * 50)
        
        metrics = self.data.agg({
            'AdSpend': 'mean',
            'Income': 'mean',
            'ConversionRate': 'mean'
        })
        
        print(f"Average Ad Spend: ${metrics['AdSpend']:,.2f}")
        print(f"Average Customer Income: ${metrics['Income']:,.2f}")
        print(f"Average Conversion Rate: {metrics['ConversionRate']:.1%}")
        
        return metrics
    
    def rfm_analysis(self):
        """Perform RFM (Recency, Frequency, Monetary) analysis"""
        print("\n" + "=" * 50)
        print("RFM SEGMENT ANALYSIS")
        print("=" * 50)
        
        rfm_summary = self.data.groupby('RFM_Segment').agg({
            'CustomerID': 'count',
            'Conversion': 'mean',
            'AdSpend': 'mean',
            'Income': 'mean'
        }).round(2)
        
        rfm_summary.columns = ['Customer_Count', 'Conversion_Rate', 'Avg_AdSpend', 'Avg_Income']
        
        print("RFM Segment Summary:")
        print(rfm_summary.to_string())
        
        # Find best performing segments
        best_conversion = rfm_summary.loc[rfm_summary['Conversion_Rate'].idxmax()]
        print(f"\nBest Converting Segment: {rfm_summary['Conversion_Rate'].idxmax()}")
        print(f"Conversion Rate: {best_conversion['Conversion_Rate']:.1%}")
        
        return rfm_summary
    
    def clv_analysis(self):
        """Analyze Customer Lifetime Value"""
        print("\n" + "=" * 50)
        print("CUSTOMER LIFETIME VALUE ANALYSIS")
        print("=" * 50)
        
        print("CLV Distribution:")
        clv_stats = self.data['CLV'].describe()
        for stat, value in clv_stats.items():
            print(f"  {stat.capitalize()}: ${value:,.2f}")
        
        # CLV by Income Segment
        print("\nCLV by Income Segment:")
        income_segments = pd.cut(self.data['Income'], bins=3)
        clv_by_income = self.data.groupby(income_segments)['CLV'].mean()
        
        for segment, avg_clv in clv_by_income.items():
            print(f"  {segment}: ${avg_clv:,.2f}")
        
        return clv_stats, clv_by_income
    
    def calculate_cpa(self):
        """Calculate Cost Per Acquisition by channel"""
        print("\n" + "=" * 50)
        print("COST PER ACQUISITION (CPA) ANALYSIS")
        print("=" * 50)
        
        def calc_cpa(group):
            return group['AdSpend'].sum() / group['Conversion'].count()
        
        cpa_by_channel = self.data.groupby('CampaignChannel').apply(calc_cpa, include_groups=False)
        
        print("CPA by Campaign Channel:")
        for channel, cpa in cpa_by_channel.items():
            print(f"  {channel}: ${cpa:,.2f}")
        
        best_cpa_channel = cpa_by_channel.idxmin()
        print(f"\nBest CPA Channel: {best_cpa_channel} (${cpa_by_channel.min():,.2f})")
        
        return cpa_by_channel
    
    def calculate_roas(self):
        """Calculate Return on Ad Spend"""
        print("\n" + "=" * 50)
        print("RETURN ON AD SPEND (ROAS) ANALYSIS")
        print("=" * 50)
        
        def calc_roas(group):
            if group['AdSpend'].sum() > 0:
                return group[group['Conversion'] == 1]['Income'].sum() / group['AdSpend'].sum()
            return 0
        
        roas_by_type = self.data.groupby('CampaignChannel').apply(calc_roas, include_groups=False)
        
        print("ROAS by Campaign Type:")
        for campaign_type, roas in roas_by_type.items():
            print(f"  {campaign_type}: {roas:.2f}x")
        
        best_roas_type = roas_by_type.idxmax()
        print(f"\nBest ROAS Campaign Type: {best_roas_type} ({roas_by_type.max():.2f}x)")
        
        return roas_by_type
    
    def campaign_effectiveness(self):
        """Analyze campaign effectiveness with composite scoring"""
        print("\n" + "=" * 50)
        print("CAMPAIGN EFFECTIVENESS ANALYSIS")
        print("=" * 50)
        
        effectiveness = self.data.groupby('CampaignType').agg({
            'ClickThroughRate': 'mean',
            'ConversionRate': 'mean',
            'AdSpend': 'sum',
            'Conversion': 'sum'
        })
        
        # Calculate efficiency (conversions per $1000 spent)
        effectiveness['Efficiency'] = effectiveness['Conversion'] / effectiveness['AdSpend'] * 1000
        
        # Composite Score (weighted average of key metrics)
        effectiveness['Composite_Score'] = (
            effectiveness['ClickThroughRate'] * 0.3 +      # Engagement weight
            effectiveness['ConversionRate'] * 0.4 +        # Conversion weight  
            effectiveness['Efficiency'] * 0.3              # Efficiency weight
        )
        
        print("Campaign Effectiveness Metrics:")
        print(effectiveness.round(4).to_string())
        
        best_campaign = effectiveness.loc[effectiveness['Composite_Score'].idxmax()]
        print(f"\nBest Overall Campaign Type: {effectiveness['Composite_Score'].idxmax()}")
        print(f"Composite Score: {best_campaign['Composite_Score']:.4f}")
        
        return effectiveness
    
    def engagement_analysis(self):
        """Analyze engagement metrics correlation"""
        print("\n" + "=" * 50)
        print("ENGAGEMENT CORRELATION ANALYSIS")
        print("=" * 50)
        
        engagement_metrics = [
            'WebsiteVisits', 'TimeOnSite', 'EmailOpens', 
            'EmailClicks', 'SocialShares', 'PagesPerVisit'
        ]
        
        # Correlation with conversion
        engagement_corr = self.data[engagement_metrics + ['Conversion']].corr()['Conversion'].drop('Conversion')
        
        print("Engagement Metrics Correlation with Conversion:")
        for metric, corr in engagement_corr.sort_values(ascending=False).items():
            print(f"  {metric}: {corr:.4f}")
        
        # CLV correlation
        clv_engagement_corr = self.data[engagement_metrics + ['CLV_Basic']].corr()['CLV_Basic'].drop('CLV_Basic')
        
        print("\nCLV Correlation with Engagement Metrics:")
        for metric, corr in clv_engagement_corr.sort_values(ascending=False).items():
            print(f"  {metric}: {corr:.4f}")
        
        return engagement_corr, clv_engagement_corr
    
    def demographic_channel_analysis(self):
        """Analyze best channels by demographic segments"""
        print("\n" + "=" * 50)
        print("DEMOGRAPHIC CHANNEL ANALYSIS")
        print("=" * 50)
        
        # Create pivot table for better visualization
        demographic_performance = self.data.groupby(['Age_Groups', 'Gender', 'CampaignChannel'])['ConversionRate'].mean().unstack('CampaignChannel')
        
        print("Conversion Rates by Demographics and Channel:")
        print(demographic_performance.round(4).to_string())
        
        # Find best channel for each demographic
        print("\nBest Channel Recommendations by Demographic:")
        for age_group in self.data['Age_Groups'].cat.categories:
            for gender in self.data['Gender'].unique():
                subset = self.data[(self.data['Age_Groups'] == age_group) & (self.data['Gender'] == gender)]
                if len(subset) > 0:
                    channel_performance = subset.groupby('CampaignChannel')['ConversionRate'].mean()
                    best_channel = channel_performance.idxmax()
                    best_rate = channel_performance.max()
                    print(f"  {age_group} {gender}: {best_channel} ({best_rate:.1%})")
        
        return demographic_performance
    
    def generate_full_report(self):
        """Generate a comprehensive marketing analysis report"""
        print("🚀 COMPREHENSIVE MARKETING CAMPAIGN ANALYSIS REPORT")
        print("=" * 80)
        
        # Run all analyses
        self.get_data_overview()
        self.analyze_campaign_channels()
        self.calculate_key_metrics()
        self.rfm_analysis()
        self.clv_analysis()
        self.calculate_cpa()
        self.calculate_roas()
        self.campaign_effectiveness()
        self.engagement_analysis()
        self.demographic_channel_analysis()
        
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 80)


def main():
    """
    Main function to run the marketing analysis
    """
    # Initialize the analyzer
    # Note: Update the path to your actual CSV file location
    data_path = r"C:\Users\Ahmed\Desktop\Data Analysis Projects\GitHup Portfolio\Marketing Project\digital_marketing_campaign_dataset.csv"
    
    try:
        analyzer = MarketingAnalyzer(data_path)
        analyzer.generate_full_report()
    except FileNotFoundError:
        print(f"Error: Could not find the data file at {data_path}")
        print("Please update the data_path variable with the correct file location.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()  