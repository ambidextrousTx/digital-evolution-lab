import streamlit as st
import time
import numpy as np
from algorithms.genetic_algorithm import GeneticAlgorithm
from viz.plotting import create_animated_heatmap, create_animated_fitness_fig

st.set_page_config(page_title="Evo Playground • Live", layout="wide")
st.title("🌿 Evo-Playground: Live Genetic Algorithm Animation")

# Controls
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Parameters")
with col2:
    speed = st.selectbox("Animation Speed", ["Fast", "Medium", "Slow", "Step-by-Step"], index=1)

params_col1, params_col2 = st.columns(2)
with params_col1:
    pop_size = st.slider("Population Size", 50, 500, 200, step=25)
    genome_len = st.slider("Genome Length", 20, 200, 100)
    mut_rate = st.slider("Mutation Rate", 0.001, 0.1, 0.015, step=0.001)
with params_col2:
    cross_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.8, step=0.05)
    elitism = st.slider("Elitism", 0, 20, 5)
    max_gens = st.slider("Max Generations", 50, 400, 200)

run_button = st.button("🚀 Start Live Evolution", type="primary", use_container_width=True)

if run_button:
    ga = GeneticAlgorithm(pop_size, genome_len, mut_rate, cross_rate, elitism)
    ga.initialize_population()

    # Create persistent placeholders (this is the key to no scrolling)
    header = st.empty()
    viz_container = st.empty()
    metrics = st.empty()
    progress = st.progress(0)
    stop_button = st.button("⏹️ Stop", key="stop_btn")

    delay = {"Fast": 0.05, "Medium": 0.15, "Slow": 0.4, "Step-by-Step": 1.0}[speed]

    for gen in range(max_gens):
        if stop_button:  # Note: this won't work perfectly in one run — we'll improve later
            break

        ga.evolve_one_generation()

        # Update every generation for smoothness (or every 2-3 if too slow)
        if gen % max(1, int(pop_size/80)) == 0 or gen == max_gens-1:
            progress.progress((gen + 1) / max_gens)

            best_fit = max(ind.fitness for ind in ga.population)
            header.markdown(f"### Generation **{gen}** — Best Fitness: **{best_fit}/{genome_len}**")

            metrics.metric("Mean Fitness", f"{np.mean([ind.fitness for ind in ga.population]):.1f}")

            # Combined animated figure
            combined_fig = create_animated_fitness_fig(ga.population, gen, ga.best_history)
            heatmap_fig = create_animated_heatmap(ga.population, gen)

            with viz_container.container():
                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(heatmap_fig, use_container_width=True, key=f"heat_{gen}")
                with col_b:
                    st.plotly_chart(combined_fig, use_container_width=True, key=f"fit_{gen}")

            time.sleep(delay)  # controls animation speed

    st.success("✅ Evolution finished! Play with parameters and run again.")
    st.balloons()
