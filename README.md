# AIM2
AIM2 Class repo


# GenAI Stability in Diffusion Models: A Dynamical Systems Approach for Medical Imaging

---

## 1. ORIGINAL IDEA

### 1.1 Core Strengths

1. **Direct Alignment with Cutting-Edge Research**: A paper just published in *Patterns* (Cell Press, December 2025) titled *"Autonomous language-image generation loops converge to generic visual motifs"* demonstrates **exactly** the phenomenon you independently conceived. They found that SDXL + LLaVA image→text→image loops converge to just 12 "visual attractors" regardless of initial prompts—what they termed "visual elevator music."

2. **Medical Domain Advantage**: Unlike the Patterns paper which used natural images, you have a **constrained, well-defined manifold** (chest X-rays). CXRs have consistent structure (lungs, heart, diaphragm), making the dynamical systems analysis more tractable and interpretable.

3. **Existing Infrastructure**: You already have:
   - Trained CLIP model for CXR-report joint embeddings
   - RAG-based retrieval system
   - Diffusion model for CXR generation
   - The loop experiment framework

4. **Novel Theoretical Framing**: Your intuition about phase planes, fixed points, isoclines, and oscillatory behavior is **exactly** the right theoretical framework, but has not been rigorously applied to medical GenAI.

### 1.2 Key Refinements Needed

Your idea needs to be **sharpened** from a general "stability analysis" to a **specific, testable hypothesis** with novel contributions beyond existing work:

| Existing Work (Patterns 2025) | Your Opportunity |
|-------------------------------|------------------|
| Natural images (SDXL/LLaVA) | Medical images (constrained manifold) |
| Qualitative attractor identification | Quantitative dynamical systems metrics |
| Descriptive (12 clusters) | Predictive (can we characterize attractors a priori?) |
| No clinical relevance | Clinical implications for AI-assisted diagnosis |

---

## 2. EXTENSIVE LITERATURE BACKGROUND

### 2.1 Model Collapse & Iterative Generation

**Foundational Work:**
- **Shumailov et al. (Nature, 2024)**: "AI models collapse when trained on recursively generated data" - Established that iterative training on synthetic data causes "model collapse" with loss of tail distributions.
- **Bertrand et al. (ICLR 2024)**: "On the stability of iterative retraining of generative models on their own data" - Showed that mixing real/synthetic data can stabilize retraining.
- **Hintze et al. (Patterns/Cell, Dec 2025)**: The most directly relevant paper—demonstrated convergence to 12 visual attractors in autonomous image-text loops.

**Key Theoretical Insights:**
- Model collapse is a **statistical phenomenon**: variance decreases across generations
- Two phases: *early collapse* (tail loss) → *late collapse* (mode collapse)
- Can be modeled as Gaussian random walk in variance space

### 2.2 Dynamical Systems Analysis of Neural Networks

**Core References:**
- **Chemnitz et al. (arXiv 2507.05164, July 2025)**: "A Dynamical Systems Perspective on the Analysis of Neural Networks" - Comprehensive framework treating neural networks as dynamical systems with Lyapunov analysis, fixed points, and bifurcation theory.
- **NeurIPS 2024**: "Back to the Continuous Attractor" - Analyzes attractor dynamics in neural networks for working memory.
- **Chang et al.**: "Neural Lyapunov Control" - Methods for constructing Lyapunov functions for neural networks.

**Relevant Concepts:**
1. **Lyapunov Exponents**: Measure sensitivity to initial conditions; positive = chaotic, negative = convergent
2. **Fixed Point Analysis**: Identifying stable/unstable equilibria in embedding space
3. **Basin of Attraction**: Set of initial conditions converging to a fixed point
4. **Bifurcation Analysis**: How system behavior changes with parameters

### 2.3 CLIP/Multimodal Embedding Spaces

**Key Papers:**
- **Chou et al. (arXiv 2409.13079, Sep 2024)**: "Embedding Geometries of Contrastive Language-Image Pre-Training" - Analyzes CLIP embedding geometry (Euclidean vs hyperbolic).
- **Liang et al. (NeurIPS 2022)**: "Mind the gap: understanding the modality gap in multi-modal contrastive representation learning" - Identifies systematic gaps between modalities in CLIP.
- **RoentGen (Nature Biomed Eng, 2024)**: Vision-language foundation model for CXR generation using MIMIC-CXR.

**Important Geometric Properties:**
- CLIP embeddings lie on a **hypersphere** (L2-normalized)
- Modality gap: text and image embeddings occupy different regions
- Cosine similarity defines "distance" in this space

### 2.4 Medical Imaging GenAI

**Foundational Work:**
- **MIMIC-CXR**: 377,110 images with free-text reports
- **Google CXR Foundation**: Embeddings for chest X-rays
- **Ark+ (Nature, June 2025)**: Foundation model for chest radiography
- **Multiple diffusion models** for CXR synthesis have been developed

### 2.5 Gap in the Literature

**What's Missing:**
1. No rigorous dynamical systems analysis of multimodal generative loops
2. No characterization of attractors in **medical** imaging domains
3. No predictive framework for attractor locations based on embedding geometry
4. No clinical validation of what "stability" means for diagnostic AI

---

## 3. PROPOSED PROJECT: DYNAMICAL ATTRACTOR ANALYSIS OF MEDICAL MULTIMODAL GENERATION

### 3.1 Project Title

**"Attractor Dynamics in Medical Vision-Language Generation: A Lyapunov Analysis of CXR-Report Loops"**

Alternative titles:
- "Fixed Points of Thought: Dynamical Systems Analysis of Medical AI Generation Loops"
- "Where Do Medical Diffusion Models Converge? Characterizing Attractors in CXR-Report Space"

### 3.2 Core Hypothesis

> **Hypothesis**: Iterative CXR-report generation loops converge to a finite set of clinical attractors whose locations in embedding space are predictable from the joint distribution geometry, and whose clinical semantics correspond to high-frequency pathological patterns (e.g., "normal chest", "cardiomegaly + effusion").

### 3.3 Research Questions

1. **RQ1**: Do medical vision-language generation loops exhibit attractor dynamics similar to natural images?
2. **RQ2**: Can we characterize these attractors using classical dynamical systems metrics (Lyapunov exponents, basin geometry)?
3. **RQ3**: What is the clinical interpretation of the attractors? (Normal vs. pathological)
4. **RQ4**: Can we predict convergence speed and attractor membership from initial embedding location?
5. **RQ5**: How do RAG retrieval augmentation affect attractor dynamics?

### 3.4 Methodology

#### Phase 1: Experimental Setup (Weeks 1-3)

**Infrastructure:**
- Use your existing CLIP model trained on MIMIC-CXR
- Diffusion model for CXR generation (fine-tuned Stable Diffusion or train from scratch)
- LLM for report generation (could use LLaVA, or medical-specific like Med-Flamingo)

**Loop Configurations:**
```
Configuration A: Image → Report → Image → Report → ... (basic loop)
Configuration B: Image → [RAG] → Report → [RAG] → Image → ... (RAG-augmented)
Configuration C: Image → Report → Image (single modality fixed) → ...
```

**Data:**
- Initialize from diverse CXR images (N=1000 trajectories)
- Stratify by: normal, common pathologies (cardiomegaly, effusion, pneumonia), rare pathologies
- Run for T=100+ iterations per trajectory

#### Phase 2: Dynamical Systems Analysis (Weeks 4-8)

**Metrics to Compute:**

1. **Trajectory Analysis in Embedding Space:**
   - Track embeddings e(t) = {e_img(t), e_text(t)} through iterations
   - Compute inter-step distance: d(t) = ||e(t+1) - e(t)||
   - Measure convergence rate: λ = lim_{t→∞} (1/t) log(d(t)/d(0))

2. **Lyapunov Exponent Estimation:**
   ```python
   def estimate_lyapunov(trajectory, perturbation_size=1e-4):
       # Perturb initial condition
       # Track divergence/convergence of trajectories
       # Compute maximum Lyapunov exponent
   ```

3. **Fixed Point Identification:**
   - Cluster final states (t=100) using k-means with elbow method
   - Identify fixed points as cluster centroids
   - Verify stability: perturb and check convergence back

4. **Basin of Attraction Mapping:**
   - For each identified attractor, map which initial conditions lead there
   - Visualize using UMAP/t-SNE with basin coloring

5. **Phase Portrait Construction:**
   - Project onto principal components of embedding space
   - Draw vector field showing flow direction
   - Identify nullclines and bifurcation points

#### Phase 3: Clinical Interpretation (Weeks 9-12)

**Clinical Attractor Characterization:**
- Extract reports/images at attractors
- Have radiologist annotate clinical content
- Correlate with ICD codes from MIMIC-IV

**Pathology-Specific Analysis:**
- Do different pathologies converge to different attractors?
- Is there "diagnostic collapse" (all → "normal")?
- How do rare diseases behave vs. common ones?

#### Phase 4: Predictive Modeling (Weeks 13-16)

**Attractor Prediction:**
- Train classifier: initial embedding → final attractor
- Features: embedding coordinates, distance to known attractors, local curvature

**Convergence Time Prediction:**
- Regression model for time-to-convergence
- Identify factors that accelerate/delay convergence

### 3.5 Novel Contributions

| Contribution | Novelty | Significance |
|--------------|---------|--------------|
| First dynamical systems analysis of medical GenAI loops | High | New theoretical framework |
| Lyapunov characterization of multimodal attractors | High | Rigorous stability metrics |
| Clinical interpretation of attractors | Medium-High | Bridges theory and practice |
| Basin of attraction mapping in CLIP space | High | Geometric understanding |
| RAG impact on attractor dynamics | Medium | Practical implications |
| Predictive model for convergence | Medium | Enables intervention design |

### 3.6 Expected Outcomes

1. **Empirical Finding**: Medical vision-language loops converge to K clinical attractors (K ≈ 10-20)
2. **Theoretical Contribution**: Lyapunov-based characterization of attractor stability
3. **Clinical Insight**: Attractors correspond to high-frequency diagnostic patterns
4. **Practical Tool**: Predictor for when a generative system is "trapped" in an attractor

### 3.7 Experimental Design Table

| Experiment | Independent Variable | Dependent Variable | Sample Size |
|------------|---------------------|-------------------|-------------|
| E1: Basic Loop | Initial image class | Final attractor, convergence time | 1000 trajectories |
| E2: Temperature | Sampling temperature (0.1-1.5) | Attractor diversity | 500 × 7 temps |
| E3: RAG Impact | With/without RAG | Convergence dynamics | 500 × 2 |
| E4: Pathology | Pathology type (6 classes) | Attractor membership | 600 |
| E5: Model Swap | Different diffusion models | Attractor stability | 300 × 3 models |

---

## 4. FEASIBILITY ANALYSIS

### 4.1 Timeline (4 months)

| Month | Activities | Deliverables |
|-------|------------|--------------|
| **Month 1** | Infrastructure setup, baseline experiments | Working loop system, initial trajectories |
| **Month 2** | Full trajectory generation, metric computation | 1000+ trajectories, Lyapunov estimates |
| **Month 3** | Attractor analysis, clinical annotation | Attractor catalog, basin maps |
| **Month 4** | Predictive modeling, paper writing | Complete analysis, draft paper |

### 4.2 Computational Requirements

- **GPU**: A100 or similar for diffusion model inference
- **Storage**: ~500GB for trajectories and embeddings
- **Compute Time**: ~200 GPU-hours for full experiment suite

### 4.3 Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| No clear attractors emerge | Low | Prior work shows this is unlikely; adjust iterations |
| Computational bottleneck | Medium | Pre-compute embeddings, use efficient sampling |
| Clinical annotation time | Medium | Start with automated CheXbert labels |
| Paper scope creep | Medium | Focus on 2-3 core findings |

---

## 5. ALTERNATIVE/COMPLEMENTARY DIRECTIONS

If you want to pivot or add components:

### 5.1 Theoretical Focus
- **Pure dynamical systems**: Prove theorems about convergence rates under specific CLIP geometry assumptions
- **Information-theoretic**: Analyze entropy reduction across iterations

### 5.2 Clinical Focus
- **Diagnostic Safety**: Do attractors represent "safe" diagnoses? Is there bias toward missing rare diseases?
- **Intervention Design**: How to "escape" a bad attractor during AI-assisted diagnosis

### 5.3 Methodological Focus
- **Novel Metrics**: Propose new stability metrics tailored for medical GenAI
- **Benchmark**: Create a benchmark for evaluating multimodal generation stability


## 6. KEY REFERENCES

### Must-Cite Papers:

1. Hintze et al. (2025). "Autonomous language-image generation loops converge to generic visual motifs." *Patterns* (Cell Press).

2. Shumailov et al. (2024). "AI models collapse when trained on recursively generated data." *Nature*.

3. Chemnitz et al. (2025). "A Dynamical Systems Perspective on the Analysis of Neural Networks." *arXiv*.

4. Bluethgen et al. (2024). "A vision-language foundation model for the generation of realistic chest X-ray images." *Nature Biomed. Eng.*

5. Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML* (CLIP).

6. Ho et al. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*.

### Additional Key References:

7. Liang et al. (2022). "Mind the gap: understanding the modality gap in multi-modal contrastive representation learning." *NeurIPS*.

8. Chang et al. (2019). "Neural Lyapunov Control." *NeurIPS*.

9. Johnson et al. (2019). "MIMIC-CXR: A de-identified publicly available database of chest radiographs with free-text reports." *Scientific Data*.

10. Chou et al. (2024). "Embedding Geometries of Contrastive Language-Image Pre-Training." *arXiv*.

