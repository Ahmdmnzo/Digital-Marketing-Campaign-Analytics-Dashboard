import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from main import MarketingAnalyzer

# Page configuration
st.set_page_config(
    layout="wide", 
    page_title='Marketing Campaign Analytics Dashboard',
    page_icon='📊',
    initial_sidebar_state='expanded'
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.tab-header {
    font-size: 1.8rem;
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 1rem;
}
.metric-container {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.insight-box {
    background-color: #e3f2fd;
    padding: 1rem;
    border-left: 4px solid #2196f3;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📊 Digital Marketing Campaign Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Comprehensive analysis of 8,000+ customer records across multiple campaign channels and demographics**")
st.markdown("---")

# Sidebar filters
st.sidebar.header("🎛️ Dashboard Controls")

# Initialize analyzer (you'll need to update the path)
@st.cache_data
def load_data():
    try:
        # Update this path to your actual CSV file location
        analyzer = MarketingAnalyzer(r"C:\Users\Ahmed\Desktop\Data Analysis Projects\GitHup Portfolio\Marketing Project\digital_marketing_campaign_dataset.csv")
        return analyzer
    except FileNotFoundError:
        st.error("Data file not found. Please update the file path in the code.")
        return None

analyzer = load_data()

if analyzer is None:
    st.stop()

# Sidebar filters
channels = st.sidebar.multiselect(
    "Select Campaign Channels", 
    options=analyzer.data['CampaignChannel'].unique(),
    default=analyzer.data['CampaignChannel'].unique()
)

campaign_types = st.sidebar.multiselect(
    "Select Campaign Types",
    options=analyzer.data['CampaignType'].unique(),
    default=analyzer.data['CampaignType'].unique()
)

age_groups = st.sidebar.multiselect(
    "Select Age Groups",
    options=analyzer.data['Age_Groups'].cat.categories,
    default=analyzer.data['Age_Groups'].cat.categories
)

# Filter data based on selections
filtered_data = analyzer.data[
    (analyzer.data['CampaignChannel'].isin(channels)) &
    (analyzer.data['CampaignType'].isin(campaign_types)) &
    (analyzer.data['Age_Groups'].isin(age_groups))
]

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    '🎯 Executive KPI Overview', 
    '📈 Campaign Performance', 
    '👥 Customer Intelligence', 
    '💰 Cost Efficiency', 
    '🔗 Engagement Analytics', 
    '🎪 Demographic Targeting'
])

# TAB 1: Executive KPI Overview
with tab1:
    st.markdown('<h2 class="tab-header">📊 Executive KPI Overview</h2>', unsafe_allow_html=True)
    
    # Calculate metrics for filtered data
    avg_ad_spend = filtered_data['AdSpend'].mean()
    avg_conversion_rate = filtered_data['ConversionRate'].mean()
    avg_income = filtered_data['Income'].mean()
    total_customers = len(filtered_data)
    total_conversions = filtered_data['Conversion'].sum()
    total_revenue = filtered_data[filtered_data['Conversion'] == 1]['Income'].sum() * 0.15  # Proxy revenue
    
    # Top row metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💸 Average Ad Spend",
            value=f"${avg_ad_spend:,.2f}",
            delta=f"{((avg_ad_spend / analyzer.data['AdSpend'].mean()) - 1) * 100:.1f}%"
        )
    
    with col2:
        st.metric(
            label="🎯 Conversion Rate",
            value=f"{avg_conversion_rate:.2%}",
            delta=f"{((avg_conversion_rate / analyzer.data['ConversionRate'].mean()) - 1) * 100:.1f}%"
        )
    
    with col3:
        st.metric(
            label="👥 Total Customers",
            value=f"{total_customers:,}",
            delta=f"{((total_customers / len(analyzer.data)) - 1) * 100:.1f}%"
        )
    
    with col4:
        st.metric(
            label="💰 Est. Revenue",
            value=f"${total_revenue:,.0f}",
            delta="Filtered Data"
        )
    
    st.markdown("---")
    
    # Performance overview charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Channel Performance Overview")
        channel_performance = filtered_data.groupby('CampaignChannel').agg({
            'ConversionRate': 'mean',
            'AdSpend': 'mean',
            'CustomerID': 'count'
        }).reset_index()
        
        fig = px.scatter(
            channel_performance,
            x='AdSpend',
            y='ConversionRate',
            size='CustomerID',
            color='CampaignChannel',
            title="Channel Performance: Conversion Rate vs Ad Spend",
            labels={'AdSpend': 'Average Ad Spend ($)', 'ConversionRate': 'Conversion Rate'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Campaign Type Distribution")
        campaign_dist = filtered_data['CampaignType'].value_counts()
        
        fig = px.pie(
            values=campaign_dist.values,
            names=campaign_dist.index,
            title="Campaign Type Distribution"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("### 🔍 Key Insights")
    best_channel = filtered_data.groupby('CampaignChannel')['ConversionRate'].mean().idxmax()
    worst_channel = filtered_data.groupby('CampaignChannel')['ConversionRate'].mean().idxmin()
    
    st.markdown(f"""
    - **Best Performing Channel**: {best_channel} with {filtered_data.groupby('CampaignChannel')['ConversionRate'].mean().max():.2%} conversion rate
    - **Opportunity Channel**: {worst_channel} has potential for optimization
    - **Total Campaign Reach**: {total_customers:,} customers across {len(channels)} channels
    - **Average Customer Value**: ${avg_income:,.0f} annual income
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: Campaign Performance
with tab2:
    st.markdown('<h2 class="tab-header">📈 Campaign Performance Analysis</h2>', unsafe_allow_html=True)
    
    # Multi-metric comparison
    st.subheader("🔄 Multi-Channel Performance Comparison")
    
    # Calculate comprehensive metrics
    campaign_metrics = filtered_data.groupby('CampaignChannel').agg({
        'ConversionRate': 'mean',
        'ClickThroughRate': 'mean',
        'AdSpend': 'mean',
        'CustomerID': 'count',
        'Conversion': 'sum'
    }).reset_index()
    
    # Calculate CPA and ROAS
    campaign_metrics['CPA'] = campaign_metrics['AdSpend'] / (campaign_metrics['Conversion'] / campaign_metrics['CustomerID'])
    campaign_metrics['ROAS'] = (campaign_metrics['Conversion'] / campaign_metrics['CustomerID']) * filtered_data['Income'].mean() * 0.15 / campaign_metrics['AdSpend']
    
    # Create subplot with multiple metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Conversion Rate by Channel', 'Click-Through Rate by Channel', 
                       'Cost Per Acquisition', 'Return on Ad Spend'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Conversion Rate
    fig.add_trace(
        go.Bar(x=campaign_metrics['CampaignChannel'], y=campaign_metrics['ConversionRate'],
               name='Conversion Rate', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Click-Through Rate
    fig.add_trace(
        go.Bar(x=campaign_metrics['CampaignChannel'], y=campaign_metrics['ClickThroughRate'],
               name='CTR', marker_color='lightgreen'),
        row=1, col=2
    )
    
    # CPA
    fig.add_trace(
        go.Bar(x=campaign_metrics['CampaignChannel'], y=campaign_metrics['CPA'],
               name='CPA', marker_color='lightcoral'),
        row=2, col=1
    )
    
    # ROAS
    fig.add_trace(
        go.Bar(x=campaign_metrics['CampaignChannel'], y=campaign_metrics['ROAS'],
               name='ROAS', marker_color='gold'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Comprehensive Channel Performance Metrics")
    st.plotly_chart(fig, use_container_width=True)
    
    # Campaign effectiveness over time simulation
    st.subheader("📅 Campaign Performance Trends")
    
    # Simulate time-based performance (since we don't have actual time data)
    np.random.seed(42)
    date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    trend_data = []
    
    for channel in channels:
        base_conversion = filtered_data[filtered_data['CampaignChannel'] == channel]['ConversionRate'].mean()
        for date in date_range:
            # Add some realistic variation
            trend_data.append({
                'Date': date,
                'Channel': channel,
                'Conversion_Rate': base_conversion + np.random.normal(0, 0.02),
                'Ad_Spend': filtered_data[filtered_data['CampaignChannel'] == channel]['AdSpend'].mean() * (1 + np.random.normal(0, 0.1))
            })
    
    trend_df = pd.DataFrame(trend_data)
    
    fig = px.line(
        trend_df, 
        x='Date', 
        y='Conversion_Rate', 
        color='Channel',
        title='Campaign Performance Trends Over Time'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance ranking table
    st.subheader("🏆 Channel Performance Ranking")
    
    ranking_df = campaign_metrics.copy()
    ranking_df['Performance_Score'] = (
        ranking_df['ConversionRate'] * 0.4 +
        ranking_df['ClickThroughRate'] * 0.3 +
        (1 / ranking_df['CPA']) * 1000 * 0.2 +  # Inverse CPA (lower is better)
        ranking_df['ROAS'] * 0.1
    )
    
    ranking_df = ranking_df.sort_values('Performance_Score', ascending=False)
    ranking_df['Rank'] = range(1, len(ranking_df) + 1)
    
    st.dataframe(
        ranking_df[['Rank', 'CampaignChannel', 'ConversionRate', 'ClickThroughRate', 'CPA', 'ROAS', 'Performance_Score']].round(4),
        use_container_width=True
    )

# TAB 3: Customer Intelligence
with tab3:
    st.markdown('<h2 class="tab-header">👥 Customer Intelligence & Segmentation</h2>', unsafe_allow_html=True)
    
    # RFM Analysis
    st.subheader("🎯 RFM Customer Segmentation")
    
    rfm_summary = filtered_data.groupby('RFM_Segment').agg({
        'CustomerID': 'count',
        'ConversionRate': 'mean',
        'AdSpend': 'mean',
        'Income': 'mean',
        'CLV': 'mean'
    }).reset_index()
    
    rfm_summary.columns = ['RFM_Segment', 'Customer_Count', 'Conversion_Rate', 'Avg_AdSpend', 'Avg_Income', 'Avg_CLV']
    
    # RFM Heatmap
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create a pivot table for heatmap
        rfm_pivot = filtered_data.pivot_table(
            values='ConversionRate',
            index='R_Score',
            columns='F_Score',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(rfm_pivot, annot=True, cmap='YlOrRd', fmt='.3f', ax=ax)
        plt.title('RFM Conversion Rate Heatmap')
        plt.ylabel('Recency Score')
        plt.xlabel('Frequency Score')
        st.pyplot(fig)
    
    with col2:
        st.subheader("🏆 Top RFM Segments")
        top_segments = rfm_summary.nlargest(5, 'Conversion_Rate')[['RFM_Segment', 'Conversion_Rate', 'Customer_Count']]
        for idx, row in top_segments.iterrows():
            st.metric(
                label=f"{row['RFM_Segment']}",
                value=f"{row['Conversion_Rate']:.2%}",
                delta=f"{row['Customer_Count']} customers"
            )
    
    # CLV Analysis
    st.subheader("💎 Customer Lifetime Value Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CLV Distribution
        fig = px.histogram(
            filtered_data,
            x='CLV',
            nbins=30,
            title='Customer Lifetime Value Distribution'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # CLV by Channel
        clv_by_channel = filtered_data.groupby('CampaignChannel')['CLV'].mean().reset_index()
        
        fig = px.bar(
            clv_by_channel,
            x='CampaignChannel',
            y='CLV',
            title='Average CLV by Campaign Channel'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer segmentation insights
    st.subheader("📊 Segment Performance Analysis")
    
    segment_performance = filtered_data.groupby(['Age_Groups', 'Gender']).agg({
        'ConversionRate': 'mean',
        'CLV': 'mean',
        'AdSpend': 'mean',
        'CustomerID': 'count'
    }).reset_index()
    
    fig = px.scatter(
        segment_performance,
        x='CLV',
        y='ConversionRate',
        size='CustomerID',
        color='Age_Groups',
        symbol='Gender',
        title='Customer Segments: CLV vs Conversion Rate',
        labels={'CLV': 'Customer Lifetime Value', 'ConversionRate': 'Conversion Rate'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# TAB 4: Cost Efficiency
with tab4:
    st.markdown('<h2 class="tab-header">💰 Cost Efficiency Analysis</h2>', unsafe_allow_html=True)
    
    # Calculate cost metrics
    cost_metrics = filtered_data.groupby('CampaignChannel').agg({
        'AdSpend': ['sum', 'mean'],
        'Conversion': 'sum',
        'CustomerID': 'count'
    }).round(2)
    
    cost_metrics.columns = ['Total_Spend', 'Avg_Spend', 'Total_Conversions', 'Total_Customers']
    cost_metrics['CPA'] = cost_metrics['Total_Spend'] / cost_metrics['Total_Conversions']
    cost_metrics['Cost_Per_Customer'] = cost_metrics['Total_Spend'] / cost_metrics['Total_Customers']
    
    # Efficiency dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="💸 Total Ad Spend",
            value=f"${cost_metrics['Total_Spend'].sum():,.0f}"
        )
    
    with col2:
        st.metric(
            label="🎯 Total Conversions",
            value=f"{cost_metrics['Total_Conversions'].sum():,.0f}"
        )
    
    with col3:
        st.metric(
            label="💰 Average CPA",
            value=f"${cost_metrics['CPA'].mean():,.2f}"
        )
    
    st.markdown("---")
    
    # Cost efficiency visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Cost Per Acquisition by Channel")
        
        fig = px.bar(
            x=cost_metrics.index,
            y=cost_metrics['CPA'],
            color=cost_metrics['CPA'],
            color_continuous_scale='RdYlGn_r',
            title='Cost Per Acquisition by Channel'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💡 Spend vs Performance")
        
        fig = px.scatter(
            x=cost_metrics['Total_Spend'],
            y=cost_metrics['Total_Conversions'],
            size=cost_metrics['CPA'],
            color=cost_metrics.index,
            labels={'x': 'Total Spend ($)', 'y': 'Total Conversions'},
            title='Spend vs Conversions (Bubble size = CPA)'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Budget optimization recommendations
    st.subheader("💡 Budget Optimization Recommendations")
    
    # Calculate efficiency scores
    cost_metrics['Efficiency_Score'] = (
        (1 / cost_metrics['CPA']) * 1000 * 0.6 +  # Lower CPA is better
        (cost_metrics['Total_Conversions'] / cost_metrics['Total_Customers']) * 100 * 0.4  # Higher conversion rate is better
    )
    
    recommendations = cost_metrics.sort_values('Efficiency_Score', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("🚀 **Increase Budget**")
        top_performer = recommendations.index[0]
        st.write(f"**{top_performer}** - Most efficient channel")
        st.write(f"- CPA: ${recommendations.loc[top_performer, 'CPA']:.2f}")
        st.write(f"- Efficiency Score: {recommendations.loc[top_performer, 'Efficiency_Score']:.1f}")
    
    with col2:
        st.warning("⚠️ **Optimize or Reduce**")
        worst_performer = recommendations.index[-1]
        st.write(f"**{worst_performer}** - Needs optimization")
        st.write(f"- CPA: ${recommendations.loc[worst_performer, 'CPA']:.2f}")
        st.write(f"- Efficiency Score: {recommendations.loc[worst_performer, 'Efficiency_Score']:.1f}")
    
    # Detailed cost table
    st.subheader("📋 Detailed Cost Analysis")
    st.dataframe(cost_metrics.round(2), use_container_width=True)

# TAB 5: Engagement Analytics
with tab5:
    st.markdown('<h2 class="tab-header">🔗 Engagement Analytics</h2>', unsafe_allow_html=True)
    
    # Engagement metrics
    engagement_cols = ['WebsiteVisits', 'TimeOnSite', 'EmailOpens', 'EmailClicks', 'SocialShares', 'PagesPerVisit']
    
    # Correlation analysis
    st.subheader("🔍 Engagement Correlation Analysis")
    
    correlation_matrix = filtered_data[engagement_cols + ['ConversionRate', 'CLV']].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    plt.title('Engagement Metrics Correlation Matrix')
    st.pyplot(fig)
    
    # Engagement performance by channel
    st.subheader("📊 Engagement Performance by Channel")
    
    engagement_by_channel = filtered_data.groupby('CampaignChannel')[engagement_cols].mean()
    
    # Create radar chart for engagement metrics
    fig = go.Figure()
    
    for channel in engagement_by_channel.index:
        values = engagement_by_channel.loc[channel].values
        # Normalize values for better visualization
        normalized_values = (values - values.min()) / (values.max() - values.min() + 1e-10)
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=engagement_cols,
            fill='toself',
            name=channel
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Engagement Metrics by Channel (Normalized)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Engagement funnel analysis
    st.subheader("🌊 Engagement Funnel Analysis")
    
    # Create engagement stages
    col1, col2 = st.columns(2)
    
    with col1:
        # Funnel data
        funnel_data = {
            'Stage': ['Website Visits', 'Email Opens', 'Email Clicks', 'Social Shares', 'Conversions'],
            'Count': [
                filtered_data['WebsiteVisits'].mean(),
                filtered_data['EmailOpens'].mean(),
                filtered_data['EmailClicks'].mean(),
                filtered_data['SocialShares'].mean(),
                filtered_data['Conversion'].mean() * 100  # Scale for visibility
            ]
        }
        
        fig = px.funnel(
            funnel_data,
            x='Count',
            y='Stage',
            title='Average Engagement Funnel'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Engagement impact on CLV
        fig = px.scatter(
            filtered_data,
            x='WebsiteVisits',
            y='CLV',
            color='CampaignChannel',
            size='TimeOnSite',
            title='Website Visits vs CLV (Size = Time on Site)'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top engagement insights
    st.subheader("💡 Key Engagement Insights")
    
    # Calculate correlations with conversion
    engagement_corr = filtered_data[engagement_cols + ['ConversionRate']].corr()['ConversionRate'].drop('ConversionRate')
    top_engagement_factor = engagement_corr.abs().idxmax()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label=f"🎯 Top Engagement Driver",
            value=top_engagement_factor,
            delta=f"{engagement_corr[top_engagement_factor]:.3f} correlation"
        )
    
    with col2:
        avg_time_on_site = filtered_data['TimeOnSite'].mean()
        st.metric(
            label="⏱️ Avg Time on Site",
            value=f"{avg_time_on_site:.1f} min"
        )
    
    with col3:
        email_engagement = (filtered_data['EmailOpens'].mean() + filtered_data['EmailClicks'].mean()) / 2
        st.metric(
            label="📧 Email Engagement Score",
            value=f"{email_engagement:.1f}"
        )

# TAB 6: Demographic Targeting
with tab6:
    st.markdown('<h2 class="tab-header">🎪 Demographic Targeting Strategy</h2>', unsafe_allow_html=True)
    
    # Demographic performance analysis
    st.subheader("👥 Performance by Demographics")
    
    # Create demographic performance matrix
    demo_performance = filtered_data.groupby(['Age_Groups', 'Gender', 'CampaignChannel']).agg({
        'ConversionRate': 'mean',
        'AdSpend': 'mean',
        'CustomerID': 'count'
    }).reset_index()
    
    # Best channel by demographic
    st.subheader("🎯 Optimal Channel by Demographic Segment")
    
    best_channels = []
    for age_group in filtered_data['Age_Groups'].cat.categories:
        for gender in filtered_data['Gender'].unique():
            segment_data = filtered_data[
                (filtered_data['Age_Groups'] == age_group) & 
                (filtered_data['Gender'] == gender)
            ]
            if len(segment_data) > 0:
                channel_performance = segment_data.groupby('CampaignChannel')['ConversionRate'].mean()
                if len(channel_performance) > 0:
                    best_channel = channel_performance.idxmax()
                    best_rate = channel_performance.max()
                    best_channels.append({
                        'Age_Group': age_group,
                        'Gender': gender,
                        'Best_Channel': best_channel,
                        'Conversion_Rate': best_rate,
                        'Customer_Count': len(segment_data)
                    })
    
    recommendations_df = pd.DataFrame(best_channels)
    
    # Visualization of recommendations
    if not recommendations_df.empty:
        fig = px.sunburst(
            recommendations_df,
            path=['Age_Group', 'Gender', 'Best_Channel'],
            values='Customer_Count',
            color='Conversion_Rate',
            color_continuous_scale='Viridis',
            title='Demographic Targeting Recommendations'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed recommendations table
    st.subheader("📋 Detailed Targeting Recommendations")
    
    if not recommendations_df.empty:
        recommendations_df['Priority'] = pd.cut(
            recommendations_df['Conversion_Rate'],
            bins=3,
            labels=['Low', 'Medium', 'High']
        )
        
        # Style the dataframe
        styled_df = recommendations_df.style.format({
            'Conversion_Rate': '{:.2%}',
            'Customer_Count': '{:,}'
        }).background_gradient(subset=['Conversion_Rate'], cmap='RdYlGn')
        
        st.dataframe(styled_df, use_container_width=True)
    
    # Age and income analysis
    st.subheader("💰 Income Distribution by Demographics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Income by age group
        fig = px.box(
            filtered_data,
            x='Age_Groups',
            y='Income',
            color='Gender',
            title='Income Distribution by Age Group and Gender'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # CLV by demographics
        clv_demo = filtered_data.groupby(['Age_Groups', 'Gender'])['CLV'].mean().reset_index()
        
        fig = px.bar(
            clv_demo,
            x='Age_Groups',
            y='CLV',
            color='Gender',
            barmode='group',
            title='Average CLV by Demographics'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Strategic recommendations
    st.subheader("🚀 Strategic Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**💎 High-Value Segments**")
        if not recommendations_df.empty:
            high_value = recommendations_df.nlargest(3, 'Conversion_Rate')
            for _, row in high_value.iterrows():
                st.write(f"• **{row['Age_Group']} {row['Gender']}**: {row['Best_Channel']}")
    with col2:
        st.warning("**⚠️ Optimization Opportunities**")
        if not recommendations_df.empty:
            low_value = recommendations_df.nsmallest(3, 'Conversion_Rate')
            for _, row in low_value.iterrows():
                st.write(f"• **{row['Age_Group']} {row['Gender']}**: Consider optimizing {row['Best_Channel']}")
    
    # Advanced demographic insights
    st.subheader("🎯 Advanced Demographic Insights")
    
    # Create age group performance comparison
    age_performance = filtered_data.groupby('Age_Groups').agg({
        'ConversionRate': 'mean',
        'AdSpend': 'mean',
        'CLV': 'mean',
        'CustomerID': 'count',
        'Income': 'mean'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance metrics by age group
        fig = go.Figure()
        
        # Add conversion rate
        fig.add_trace(go.Bar(
            name='Conversion Rate (%)',
            x=age_performance['Age_Groups'],
            y=age_performance['ConversionRate'] * 100,
            yaxis='y',
            marker_color='lightblue'
        ))
        
        # Add CLV on secondary axis
        fig.add_trace(go.Scatter(
            name='CLV ($)',
            x=age_performance['Age_Groups'],
            y=age_performance['CLV'],
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='red', width=3)
        ))
        
        # Create secondary y-axis
        fig.update_layout(
            title='Age Group Performance: Conversion Rate vs CLV',
            xaxis_title='Age Groups',
            yaxis=dict(
                title='Conversion Rate (%)',
                side='left'
            ),
            yaxis2=dict(
                title='Customer Lifetime Value ($)',
                side='right',
                overlaying='y'
            ),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gender performance comparison
        gender_performance = filtered_data.groupby('Gender').agg({
            'ConversionRate': 'mean',
            'AdSpend': 'mean',
            'CLV': 'mean',
            'CustomerID': 'count'
        }).reset_index()
        
        fig = px.bar(
            gender_performance,
            x='Gender',
            y=['ConversionRate', 'AdSpend', 'CLV'],
            title='Performance Metrics by Gender',
            barmode='group'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Channel preference heatmap
    st.subheader("🔥 Channel Preference Heatmap")
    
    # Create channel preference matrix
    channel_demo_matrix = filtered_data.groupby(['Age_Groups', 'Gender', 'CampaignChannel']).size().unstack(fill_value=0)
    
    # Calculate percentages
    channel_demo_pct = channel_demo_matrix.div(channel_demo_matrix.sum(axis=1), axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        channel_demo_pct, 
        annot=True, 
        fmt='.1f', 
        cmap='YlOrRd', 
        ax=ax,
        cbar_kws={'label': 'Percentage (%)'}
    )
    plt.title('Channel Preference by Demographics (%)')
    plt.ylabel('Age Group & Gender')
    plt.xlabel('Campaign Channel')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Demographic ROI analysis
    st.subheader("💰 ROI Analysis by Demographics")
    
    # Calculate ROI for each demographic segment
    roi_analysis = []
    for age_group in filtered_data['Age_Groups'].cat.categories:
        for gender in filtered_data['Gender'].unique():
            segment_data = filtered_data[
                (filtered_data['Age_Groups'] == age_group) & 
                (filtered_data['Gender'] == gender)
            ]
            if len(segment_data) > 0:
                total_spend = segment_data['AdSpend'].sum()
                total_conversions = segment_data['Conversion'].sum()
                avg_clv = segment_data['CLV'].mean()
                estimated_revenue = total_conversions * avg_clv
                roi = (estimated_revenue - total_spend) / total_spend * 100 if total_spend > 0 else 0
                
                roi_analysis.append({
                    'Demographic': f"{age_group} {gender}",
                    'Total_Spend': total_spend,
                    'Conversions': total_conversions,
                    'Avg_CLV': avg_clv,
                    'Estimated_Revenue': estimated_revenue,
                    'ROI_Percentage': roi,
                    'Customer_Count': len(segment_data)
                })
    
    roi_df = pd.DataFrame(roi_analysis)
    
    if not roi_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # ROI comparison chart
            fig = px.bar(
                roi_df.sort_values('ROI_Percentage', ascending=True),
                x='ROI_Percentage',
                y='Demographic',
                color='ROI_Percentage',
                color_continuous_scale='RdYlGn',
                title='ROI by Demographic Segment (%)'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Spend vs Revenue scatter
            fig = px.scatter(
                roi_df,
                x='Total_Spend',
                y='Estimated_Revenue',
                size='Customer_Count',
                color='ROI_Percentage',
                hover_data=['Demographic', 'Conversions'],
                title='Spend vs Revenue by Demographic',
                color_continuous_scale='RdYlGn'
            )
            fig.add_shape(
                type="line",
                x0=0, y0=0,
                x1=roi_df['Total_Spend'].max(),
                y1=roi_df['Total_Spend'].max(),
                line=dict(dash="dash", color="gray"),
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed ROI table
    st.subheader("📊 Detailed ROI Analysis Table")
    
    if not roi_df.empty:
        # Format the dataframe for display
        display_roi_df = roi_df.copy()
        display_roi_df['Total_Spend'] = display_roi_df['Total_Spend'].apply(lambda x: f"${x:,.2f}")
        display_roi_df['Avg_CLV'] = display_roi_df['Avg_CLV'].apply(lambda x: f"${x:,.2f}")
        display_roi_df['Estimated_Revenue'] = display_roi_df['Estimated_Revenue'].apply(lambda x: f"${x:,.2f}")
        display_roi_df['ROI_Percentage'] = display_roi_df['ROI_Percentage'].apply(lambda x: f"{x:.1f}%")
        
        # Color coding for ROI
        def color_roi(val):
            if val.endswith('%'):
                num_val = float(val[:-1])
                if num_val > 50:
                    return 'background-color: lightgreen'
                elif num_val > 0:
                    return 'background-color: lightyellow'
                else:
                    return 'background-color: lightcoral'
            return ''
        
        styled_roi_df = display_roi_df.style.applymap(color_roi, subset=['ROI_Percentage'])
        st.dataframe(styled_roi_df, use_container_width=True)
    
    # Final recommendations section
    st.subheader("🎯 Final Strategic Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("**🚀 Prioritize These Segments**")
        if not roi_df.empty:
            top_roi = roi_df.nlargest(2, 'ROI_Percentage')
            for _, row in top_roi.iterrows():
                st.write(f"• **{row['Demographic']}**")
                st.write(f"  ROI: {row['ROI_Percentage']:.1f}%")
                st.write(f"  Customers: {row['Customer_Count']:,}")
    
    with col2:
        st.info("**📈 Growth Opportunities**")
        if not roi_df.empty:
            growth_opps = roi_df[(roi_df['ROI_Percentage'] > 0) & (roi_df['Customer_Count'] > roi_df['Customer_Count'].median())]
            if len(growth_opps) > 0:
                for _, row in growth_opps.head(2).iterrows():
                    st.write(f"• **{row['Demographic']}**")
                    st.write(f"  Large market with positive ROI")
                    st.write(f"  {row['Customer_Count']:,} customers")
    
    with col3:
        st.warning("**⚠️ Requires Attention**")
        if not roi_df.empty:
            low_roi = roi_df.nsmallest(2, 'ROI_Percentage')
            for _, row in low_roi.iterrows():
                st.write(f"• **{row['Demographic']}**")
                st.write(f"  ROI: {row['ROI_Percentage']:.1f}%")
                st.write(f"  Needs strategy revision")
    
    # Executive summary
    st.markdown("---")
    st.subheader("📋 Executive Summary")
    
    if not roi_df.empty and not recommendations_df.empty:
        # Calculate summary statistics
        total_customers = filtered_data['CustomerID'].nunique()
        total_spend = filtered_data['AdSpend'].sum()
        total_conversions = filtered_data['Conversion'].sum()
        avg_roi = roi_df['ROI_Percentage'].mean()
        best_demographic = roi_df.loc[roi_df['ROI_Percentage'].idxmax(), 'Demographic']
        best_roi = roi_df['ROI_Percentage'].max()
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown(f"""
            **📊 Campaign Overview:**
            - Total Customers Analyzed: {total_customers:,}
            - Total Ad Spend: ${total_spend:,.2f}
            - Total Conversions: {total_conversions:,}
            - Overall Conversion Rate: {(total_conversions/total_customers)*100:.1f}%
            """)
        
        with summary_col2:
            st.markdown(f"""
            **🎯 Key Findings:**
            - Best Performing Demographic: **{best_demographic}**
            - Highest ROI: **{best_roi:.1f}%**
            - Average ROI Across Segments: **{avg_roi:.1f}%**
            - Segments with Positive ROI: **{len(roi_df[roi_df['ROI_Percentage'] > 0])}** out of {len(roi_df)}
            """)
        
        # Action items
        st.markdown("### 🎯 Immediate Action Items")
        st.markdown("""
        1. **Scale Up**: Increase budget allocation to top-performing demographic segments
        2. **Optimize**: Review and adjust targeting strategies for underperforming segments  
        3. **Test**: Implement A/B tests for channel optimization in growth opportunity segments
        4. **Monitor**: Set up automated alerts for ROI thresholds and performance metrics
        5. **Refine**: Use insights to create more targeted persona-based campaigns
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>📊 Digital Marketing Campaign Analytics Dashboard | Built with Streamlit & Plotly</p>
    <p>🔍 For detailed analysis and insights, explore all tabs above</p>
</div>
""", unsafe_allow_html=True)