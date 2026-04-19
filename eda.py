import seaborn as sea
import matplotlib.pyplot as plot
import logging as logger

logger.basicConfig(filename="pipeline.log",level=logger.INFO)
def run_eda(df):
    logger.info("EDA Started...")

    plot.figure(figsize=(12, 8))

    sea.heatmap(df.corr(),
                 annot=True,
                 fmt=".2f",
                 cmap="coolwarm",
                 linewidths=0.5)
    
    plot.xticks(rotation=30, ha='right')
    plot.yticks(rotation=0)
    plot.xlabel("Features")
    plot.ylabel("Features")

    plot.title("Feature Correlation Matrix for Loan Prediction", fontsize=14)

    plot.savefig("correlation.png", dpi=300)
    plot.clf()

    plot.figure(figsize=(6,4))

    sea.countplot(x="Loan_Status", data=df)

    plot.title("Loan Status Distribution")
    
    for p in plot.gca().patches:
        plot.gca().annotate(f'{int(p.get_height())}',
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha='center', va='bottom')
        
    plot.savefig("loan_status.png", dpi=300)

    plot.clf()

    logger.info("EDA Completed")