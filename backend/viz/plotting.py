import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List
from core.individual import Individual


def create_population_heatmap(population: List[Individual],
                              title: str = "Population Genomes") -> go.Figure:
    """Genome heatmap: each row is one individual, each column is a gene."""
    genomes = np.array([ind.genome for ind in population])
    fig = go.Figure(data=go.Heatmap(
        z=genomes,
        colorscale=[[0, 'black'], [1, 'lime']],  # classic binary feel
        showscale=False,
        hoverongaps=False
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="Gene Position (Locus)",
        yaxis_title="Individual (sorted by fitness)",
        height=600,
        width=900,
        template="plotly_dark"
    )
    # Sort rows by fitness descending for nicer view
    fig.data[0].z = genomes[np.argsort([ind.fitness for ind in population])[::-1]]
    return fig


def create_fitness_distribution(population: List[Individual],
                                generation: int) -> go.Figure:
    """Fitness histogram + statistics."""
    fitnesses = [ind.fitness for ind in population]
    best = max(fitnesses)
    mean = np.mean(fitnesses)
    median = np.median(fitnesses)

    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Histogram(
        x=fitnesses,
        nbinsx=30,
        name="Fitness Distribution",
        marker_color="#00ff9f"
    ))

    # Vertical lines for statistics
    fig.add_vline(x=best, line_dash="dash", line_color="gold", annotation_text=f"Best: {best}")
    fig.add_vline(x=mean, line_dash="dot", line_color="white", annotation_text=f"Mean: {mean:.1f}")
    fig.add_vline(x=median, line_dash="dot", line_color="green", annotation_text=f"Median: {median:.1f}")

    fig.update_layout(
        title=f"Fitness Distribution — Generation {generation}",
        xaxis_title="Fitness (Number of 1s)",
        yaxis_title="Count",
        template="plotly_dark",
        height=400,
        showlegend=False
    )
    return fig


def create_convergence_plot(best_history: List[float],
                            title: str = "Convergence") -> go.Figure:
    """Best fitness over generations."""
    generations = list(range(len(best_history)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=generations,
        y=best_history,
        mode='lines+markers',
        line=dict(color='#00ff9f', width=3),
        marker=dict(size=4),
        name="Best Fitness"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Generation",
        yaxis_title="Best Fitness",
        template="plotly_dark",
        height=400,
        yaxis=dict(range=[0, best_history[-1] * 1.05] if best_history else None)  # assuming OneMax
    )
    return fig
