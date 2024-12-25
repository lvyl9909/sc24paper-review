---
theme: ./
layout: cover
class: text-left
backgroud: '/ATLAS/UNT-icon.png'
authors:  # First author should be the presenter
  - Mekena Metcalf: ["Innovation and Ventures HSBC Holdings Plc."]
  - Pablo Andr´es-Mart´ınez: ["Quantinuum"]
  - Nathan Fitzpatrick: ["Quantinuum"] 

meeting: "Presenter: Yilin Lyu"
preTitle: "SC24 Paper: Realizing Quantum Kernel Models at Scale with Matrix Product State Simulation"
---

<br>

[//]: # (<p style="color:#0FA3B1;">Don't explicitly put title on cover page 🥳 </p>)

[//]: # (<p style="color:#0FA3B1;">Put your own logo somewhere </p>)

<img id="ATLAS" src="/ATLAS/UNT-icon.png"> </img>

<style scoped>
#ATLAS {
  width: 180px;
  position: absolute;
  right: 3%;
  bottom: 4%;
  /* background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 15%, #146b8c 50%); */
}
</style>

---
layout: pageBar
---

# Key Contributions

Developed a quantum kernel framework using a Matrix Product State (MPS) simulator
and employ it to perform a classification task with 165 features and 6400 training data points

<br>

- 📝 **Tensor Network Framework for Quantum Kernels** - Achieved a tensor network framework using MPS to simulate quantum kernel methods.
- 🎨 **Framework Optimization** - Made use of the SVD truncation mechanism and parallel computing strategy.
- 🤹 **GPU vs CPU** - Assessed and compared the MPS simulation performance on CPU and GPU.
- 🧑‍💻 **Application in Large-scale Dataset** - Performed a classification task with 165 features and 6400 training data points, thus verifying
the quantum model performance at large scale.


<br>

<!--
well beyond the scale of any prior work
-->

---
layout: pageBar
---

# Background Knowledge

## Classical Kernels vs Quantum Kernels
<br>

### **Classical Kernel Methods**
- **Definition**: Classical kernel methods map data from a low-dimensional space to a high-dimensional feature space to enhance the performance of linear classifiers like Support Vector Machines (SVM).

<div style="text-align: center;">
    <img src="/ATLAS/kernel.png" alt="Classical Kernel Example" style="width: 800px; height: 150px;">
</div>

### **Quantum Kernel Methods**
- **Definition**: Quantum kernel methods use quantum circuits to embed data into a high-dimensional quantum feature space and calculate similarity (e.g., overlap) between quantum states.
- **Core Differences**:
    - **Quantum Feature Mapping**: Encodes data into quantum states.
    - **Quantum Kernel Function**: Measures similarity through quantum state overlaps.
- **Valuable Scenario**: When dataset is high-dimensional, complex data distribution and highly nonlinear relationships between features.
- **Key limitations**: Quantum hardware has significant noise and scale limitations.

<!--
They are able to map the nonlinear relationship of data in the original space to the linear relationship in the higher dimensional space

In quantum computing, quantum gate operations are used to implement the process of mapping classical data to quantum states

Once we get the kernel function, we can use it into the SVM algorithm to do classification or regression tasks.

Especially when the data distribution is complex, the dimensionality is high, or there are highly nonlinear relationships between features, classical kernel functions are limited in their ability to construct high-dimensional feature spaces. In contrast, quantum kernel methods can better capture complex nonlinear patterns in the data.

MPS simulation has higher accuracy and controllability, and is more suitable for large-scale experiments and theoretical verification.
-->

---
layout: pageBar
---

# Background Knowledge

## Matrix Product States (MPS)
<br>

### **Why MPS?**
- **High cost of high-dimensional tensors**: The memory and computation cost of storing and accessing elements of tensors with increasing dimension grows exponentially.

<div style="text-align: center;">
    <img src="/ATLAS/MPS.png" alt="MPS" style="width: 600px; height: 300px;">
</div>

### **What did MPS do**
MPS provides a more space-efficient way of breaking the large table (high-dimensional tensors) into a set of "pieces" (matrix) and then assembling them.

<!--
The representation of data in quantum state space provides a new function space for machine learning tasks, but the benchmarking of quantum machine learning algorithms is limited by simulation methods. Previous benchmarking of quantum kernel methods was constrained by feature dimension and data dimension, leading to inconsistent results.
In this slide, we first introduce MPS, and we will elaborate the technical stuff about tensor network and MPS later.
-->

---
layout: pageBar
---

# Methodology

Developed a quantum kernel framework using a Matrix Product State (MPS) simulator in four steps.

<br>

1️⃣ **Quantum Kernel Method** - Uncovering the principles of the quantum kernel method.
<br>
2️⃣ **Quantum Circuit Simulation** - Using MPS to simulate quantum feature mapping and kernel computation.
<br>
3️⃣ **Circuit ansatz** - Determining the kernel function through self-defined parameters and quantum circuit.
<br>
4️⃣ **Parallelization** - Proposing two parallel strategies to improve computation efficient.

---
layout: pageBar
---

# Quantum Kernel Method

The inner product, i.e., kernel function, is the key!


Classical data is mapped to quantum states, and quantum kernels are computed from their inner product squared modulus.

$$
K_{ij} = |\langle \psi(x_i), \psi(x_j) \rangle|^2
$$

$$
\textcolor{red}{\longleftarrow} \ |\psi(x)\rangle = U(x)|+\rangle^m
$$

$$
\textcolor{red}{\longleftarrow} \ U(x) = \left(e^{-iH_{XX}(x)} \cdot e^{-iH_Z(x)}\right)^r
$$

<div class="grid grid-cols-3 gap-5 items-center justify-center">

<div class="col-span-2">

> - <span style="color: #9d6fa5"> $K_{ij}$: The Gram matrix entry, which measures the similarity between data points $x_i$ and $x_j$ in the quantum feature space. </span>
> - <span style="color: #c90024"> $|\psi(x)\rangle$: The quantum state obtained by applying the unitary operator $U(x)$ to the uniform superposition state $|+\rangle^m$. </span>
> - <span style="color: #296b4c"> $U(x)$: The parameterized unitary operator consisting of single-qubit ($H_Z$) and two-qubit ($H_{XX}$) Hamiltonian interactions, repeated $r$ times. </span>
> - <span style="color: #4d45cc"> $H_Z(x) = \gamma \sum_{i=1}^m x_i \sigma_Z^i$, $H_{XX}(x) = \frac{\gamma^2 \pi}{2} \sum_{(i,j) \in G} (1 - x_i)(1 - x_j) \sigma_X^i \sigma_X^j$ </span>
> - <span style="color: #9d6fa5"> $\gamma$: A real coefficient that controls the kernel bandwidth. </span>
> - <span style="color: #c90024"> $r$: A tunable parameter indicating the number of layers in the quantum circuit. </span>

</div>
<div class="col-span-1">

<Transform :scale="1.0">
<img src="https://www.quantumdiaries.org/wp-content/uploads/2011/06/cernmug.jpg"/>
</Transform>

</div>

</div>

<style scoped>
.slidev-layout blockquote {
  font-size: 1rem;
}

li {
  margin-top: 0.25rem;
  margin-bottom: 0.25rem;
}

</style>

<!--
To begin with, quantum kernel methods map classical data into a quantum feature space by encoding it as quantum states. Once mapped, the similarity between data points is calculated using their inner product squared modulus, which we call the quantum kernel function. This similarity is represented in the Gram matrix Kij.
To get Kij, we need psi, The state is first initialised in the uniform superposition |+⟩m, that is, providing a balanced starting point. Then, the unitary operator U(x) rotates the starting point to get the unique quantum state within the high-dimensional Hilbert space.
To get psi, we need U(x), which is the quantum unitary operator. It consists of single-qubit and two-qubit interaction Hamiltonians. Single-qubit Hamiltonians describe the self-rotation of one quantum, gamma is a tunable parameter, if we consider the quantum in a bloch sphere, gamma controls the rotation intensity. Two-qubit interaction
Hamiltonians introduce the quantum entanglements, it is necessary. Once we get U(x), we can deduce the inner product Kij upward and then we get the Gram matrix to represent quantum kernel function.
-->

---
layout: pageBar
---

# Quantum Circuit Simulation

Classical simulation using state vector has too much computing overhead, so the author use tensor network here.

## Tensor Network

Tensors are multidimensional arrays. We refer to each of the axes of the array as a bond.

<div style="display: flex; justify-content: center; align-items: center; height: 100%;">
    <img src="/ATLAS/tensor.png" alt="MPS" style="width: 400px; height: 90px;">
</div>

### Tensor Contraction
Two tensors can be contracted together along a common bond.

$$
C_{abxyz} = \sum_{s=0}^{\chi_s-1} A_{abs} \cdot B_{sxyz}
$$

### Tensor Reshape
Convert a multidimensional tensor to a two-dimensional matrix, to apply decomposition algorithm like SVD.
$$
M[i][j] = C_{abxyz} \quad \text{where} \quad
i = a + b \cdot \chi_a, \quad
j = x + (y + z \cdot \chi_y) \cdot \chi_x
$$

<!--
Now we know the basic workflow of quantum kernel method, next we need to simulate the quantum circuit. We know the state of an m-qubit quantum computer can be described
by a vector of 2^m complex entries. So classical method.. Tensor network depends on the degree of entanglement in the quantum state, not the number of qubits. For large numbers of qubits and shallow circuits with low entanglement, tensor networks can be efficiently simulated.
Two important operations are tensor contraction and tensor reshape, contraction helps us merge two tensors into one, similar to the matrix multiplication, in my opinion, tensor contraction has two important effects, one is compute the inner product we just mentioned, the tensor network of two quantum states is shrunk to a scalar, so in tensor network,
tensor contraction can also be regarded as the inner product computation in kernel method. Another is simulating the quantum gate, like when a quantum gate acts on some qubits, it is necessary to contract the corresponding tensor with the tensor of the gate. In conclusion, it helps us simplify the network structure and reduce the computation overhead. Reshape helps us convert 3D tensor to 2D matrix,
the matrix-based structure allows us to do the following operations like SVD easier.
-->

---
layout: pageBar
---

# Quantum Circuit Simulation

MPS is a linear chain tensor network structure for quantum states with low entanglement.

## MPS (Matrix Product State)

Tensors are arranged in a one-dimensional chain, where each tensor is connected only to its two neighbors via virtual bonds.
<div style="display: flex; justify-content: center; align-items: center; height: 100%;">
    <img src="/ATLAS/MPS-structure.png" alt="MPS-s" style="width: 500px; height: 85px;">
</div>

### Quantum Gate Application in MPS
<div style="display: flex; justify-content: center; align-items: center; height: 100%;">

<div style="flex: 1;">
    <ul>
        <li><strong>Single-qubit gate :</strong>  Single-qubit gates are contracted with the corresponding site tensor.</li>
        <li><strong>Two-qubit gate :</strong> 
            <ul>
                <li>Step 1: The two-qubit gate G is contracted with the virtual bond of tensors T_1 and T_2, generating a new tensor T_new.</li>
                <li>Step 2: Apply SVD to T_new, resulting in U, S(singular value), and V.</li>
                <li>Step 3: Merge the diagonal matrix S into U or V, producing updated tensors T'_1 and T'_2.</li>
            </ul>
        </li>
    </ul>
</div>

<div style="flex: 1;">
    <img src="/ATLAS/MPS-operations.png" alt="MPS-o" style="width: 500px; height: 300px;">
</div>

</div>

<!--
Qubits in MPS structure are arranged in the form of linear chains. Bonds connecting tensors together are known as virtual bonds. One tensor represents one qubits here.
Think what limitation this kinda structure bring to us, first, tensor network has already suffer from the entaglement, it has high computation
complexity when we have high entanglement in the quantum circuit. now, MPS is a special structure further amplified the shortcoming, we can see, MPS can only directly capture entanglement between neighbor qubits.
If we want to describe the entanglement between non-neighbor qubits, we need to pass indirectly through multiple virtual bonds and bond dimensions, which means the exponential growth computation overhead.
Anyway, we just the reviewer of the paper. After simulating the qubit by tensor, we need to simulate the quantum gate, ..., SVD is a stable matrix-based decomposition technical, that's why we need tensor reshape.
The updated tensor retains the effect of two-qubit gate operation while controlling the bond dimension within a reasonable range. Additionally, the paper also introduce a truncation mechanism, drop the small singular value, coz it 
may do not have contribution, in this way ,we can simplify the bond dimension and computation overhead. However, the truncation error is set to less than the machine precision and I think it has little influence on the simulation results.
-->

---
layout: pageBar
---

# Circuit Ansatz

<div style="display: flex; justify-content: center; align-items: center; height: 100%;">
    <img src="/ATLAS/ansatz.png" alt="MPS-s" style="width: 1200px; height: 250px;">
</div>

- **Circuit Ansatz Goals**:
Construct quantum states to capture the data feature, optimizing circuit performance by tuning parameters r, d, and γ.

$$\ |\psi(x)\rangle = U(x)|+\rangle^m$$
$$\ U(x) = \left(e^{-iH_{XX}(x)} \cdot e^{-iH_Z(x)}\right)^r$$

- **Key Design**:
  1. Use single-qubit R_Z gates for local mapping of data features.
  2. Use two-qubit R_XX gates to capture global dependencies.
  3. Different d: d=1 means the neighbor quantum interaction, we minimize resource overhead with SWAP gates when d>1.

  
<!--
circuit ansatz is the specific design of quantum circuit. psix is the quantum state, U(x) is parameterized combinations of single-qubit and two-qubit quantum operations,
HZ and Hxx here, represents the single-qubit gate and two-qubit gate respectively, we can see in the figure a, the cyan block is the single-qubit rotation gate for each qubit,
and the right red big block represents the two-qubit interactions, will be achieved by a series of two-qubit gate. One thing I need to mention is that the parameter d, which indicates
the quantum interaction distance. For instance, when d=1, only neighbor qubit will be performed by the gate, when d=2, gate will perform with no.1 qubit and no.3 qubit. But we know in MPS, we can only
perform two-qubbit gate on neighbor qubits, so the author propose using SWAP gate first, switching the position of no.3 qubit to no.2 qubit, and then perform gate operation, and then switch back to the 
previous postion. It seems a smart method, however, I found high interaction distances (d > 1) significantly increase the number of SWAP gates, resulting in greater circuit depth and simulation complexity.
So it also leads to the worse performance of tensor network simulation.

-->

---
layout: pageBar
---

# Parallelization

Two parallel computing strategies (No-Messaging and Round-Robin) are discussed in this paper for efficiently computing each element of a Gram matrix. These strategies are based on dividing the matrix into small tiles, which are then distributed to different parallel processes.
$$
K_{ij} = |\langle \psi(x_i), \psi(x_j) \rangle|^2
$$

<div style="display: flex; justify-content: center; align-items: center; height: 100%;">
    <img src="/ATLAS/parallelization.png" alt="MPS-s" style="width: 1200px; height: 300px;">
</div>

- **Two parallelization strategies**:
  1. No-messaging strategy
  2. Round-Robin strategy


<!--
Now we know how to get the psix through a circuit ansatz, go back to the kernel function, we still need to compute all
psi for each data point. Then we compute the inner product to get Kij, as the entries of Gram matrix. Because the computation
process of Kij is independent, so the paper provides 2 parallelization strategies. The first one is no-messaging strategy,
in this method, we divide the Kij matrix into multiple square tiles. Each process is responsible for one tile. Each process is 
parallel running. In this way, we can combine the result from each process to get the final result in a shorter time. Second strategy 
is round-robin. Consider did we waste something just now? Yes, in process0 and 1, we compute the vertical psi twice. So based on it,
the author propose the round-robin. In this method, we only need to simulate each psi once, and then send the psi to other process, to
compute the inner product. This method avoids the repeating computation, but increases the communication costs, but compared to 
simulation costs, it's pretty small.
-->

---
layout: pageBar
---

# Experiment Result

## GPU advantage and resource scaling

<div style="display: flex; justify-content: center; align-items: center; height: 100%;">
    <img src="/ATLAS/experiment-d.png" alt="experiment-d" style="width: 900px; height: 300px;">
</div>

### Results:
- GPUs are more efficient for highly entangled systems.
- CPU advantages in low-entanglement systems due to low initialization overhead.
### Conclusions:
- For circuits with low interaction distance, it is recommended to prioritize using CPUs (ITensors backend).
- Circuits with high interaction distance (e.g., d<9), GPUs are more effective for handling high-dimensional entanglement.

<!--
Now we know all the methods in this paper, next we will move to the experiment result part. The paper mainly has two parts,
the first part is exploring the effect of interaction distance d and gamma on performance through GPU and CPU respectively.
We can see the following two plots. We can see when d<9, CPU shows the shorter simulation time and inner product computation time,
For higher entangled systems, which means the higher interaction distance, GPU is more efficient. Based on the results, the author
throw two conclusions. This experiment is also an extra work for this paper, because in reality, we don't need to know which GPU or
CPU performs well and which performs badly in this approach. Another interesting point is that this paper spent about 500 words in
the introduction to explain a newly published tensor network paper similar to this paper. In that paper, the author explained that 
GPU is always better than CPU in tasks, so the experiment was re-conducted in this paper. The CPU is superior to the GPU in certain conditions.
-->

---
layout: pageBar
---

# Experiment Result

## Quantum Machine Learning Performance
<br>

### Experiment Goals:
- To explore classification capabilities at data volumes and feature dimensions through the simulator framework.
- To assess how the interaction distance or number of circuit layers affects the model’s classification performance.

<br>

### Experiment Desgin
- Evaluation Metrics: Accuracy, Recall, Precision, AUC.
- Variables: Data sample sizes, features, interaction distance d, coefficients γ.
- Baselines: The common Gaussian kernel, which is defined as:
  $$
  K(x, x') = e^{-\alpha \|x - x'\|^2}
  $$

---
layout: pageBar
---

# Experiment Result

## Quantum Machine Learning Performance
<div style="display: flex; justify-content: space-between; align-items: center; height: 100%;">
    <img src="/ATLAS/experiment-2.png" alt="experiment-2" style="width: 54%; height: 400px;">
    <img src="/ATLAS/experiment-2-1.png" alt="experiment-2-1" style="width: 44%; height: 400px;">
</div>

### Results:
- When the parameters are chosen appropriately (e.g. γ = 0.5, d = 4), quantum kernel outperform Gaussian kernel.
- After using quantum kernel, the model can maintain good classification performance in high feature dimension and large data scale.

<!--
Now we know all the methods in this paper, next we will move to the experiment result part. The paper mainly has two parts,
the first part is exploring the effect of interaction distance d and gamma on performance through GPU and CPU respectively.
We can see the following two plots. We can see when d<9, CPU shows the shorter simulation time and inner product computation time,
For higher entangled systems, which means the higher interaction distance, GPU is more efficient. Based on the results, the author
throw two conclusions. This experiment is also an extra work for this paper, because in reality, we don't need to know which GPU or
CPU performs well and which performs badly in this approach. Another interesting point is that this paper spent about 500 words in
the introduction to explain a newly published tensor network paper similar to this paper. In that paper, the author explained that 
GPU is always better than CPU in tasks, so the experiment was re-conducted in this paper. The CPU is superior to the GPU in certain conditions.
-->

---
layout: pageBar
---

# Limitations and Potentials

<div style="display: flex; justify-content: center; align-items: center; height: 100%;">
    <div style="width: 80%; background: #f4f4f4; padding: 20px; border-radius: 8px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);">
        <ul style="line-height: 1.8;">
            <li>
                <strong style="color: #4A90E2;">Exploring more flexible circuit structures:</strong>
                The paper adopts a <strong>linear chain </strong> topology, but introducing more complex structures, such as 2D grid, fully connected topologies, could enhance connectivity between qubits.
            </li>
            <li>
                <strong style="color: #50C878;">Adaptive parameter adjustment strategies:</strong>
                In experiment stage, key parameters, such as coefficients γ, circuit layers r and interaction distance d, are currently <strong>selected manually</strong>. Developing adaptive algorithms, such as gradient-based optimization or reinforcement learning, could dynamically find optimal parameter combinations to enhance performance and reduce overfitting risks.
            </li>
            <li>
                <strong style="color: #FF8C00;">More precise error analysis and control:</strong>
                Current SVD truncation mechanism is relatively <strong>strict</strong>, further investigation into how truncation errors affect quantum kernel model performance is essential, that is, a guaranteed and aggressive truncation mechanism. Experiments across different truncation thresholds could identify more effective strategies for balancing accuracy and efficiency.
            </li>
            <li>
                <strong style="color: #D32F2F;">Noise simulation and compensation:</strong>
                Although the paper focuses on <strong>noiseless simulations</strong>, real quantum systems inevitably involve noise. Incorporating noise models, such as decoherence and gate operation inaccuracies, and exploring noise mitigation techniques like quantum error correction or noise-adaptive algorithms could improve robustness in noisy environments.
            </li>
        </ul>
    </div>
</div>

<!--
1. As we mentioned before, Linear chain further amplified the shortcoming when we face to high-entanglement system. A better topology might improve the model's ability to capture data features, especially when we have lots of two-qubit interactions.
3. current one only cut the pretty small singular value, save a lot of values, which leads to a high virtual bond dimension and high resource overheads. Finding a better truncation value provides us a trade-off between performance and resource costs.
-->

---
layout: pageBar
---

# Conclusions
<br>

<div style="display: flex; justify-content: left; align-items: center; height: 100%; padding: 20px; background-color: #f9f9f9; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2); font-size: 1.5rem; line-height: 1.8; text-align: left;">

<p>
    This paper proposed a <span style="color: #4d79ff; font-weight: bold;">tensor network simulation framework</span> effectively expanding quantum kernel methods. It identified the <span style="color: #ff4d4d; font-weight: bold;">advantages of CPU and GPU</span> in different circuit complexities and demonstrated the <span style="color: #4caf50; font-weight: bold;">performance improvement potential</span> of quantum kernel methods on large-scale data.
    For future work, we consider the application of more <span style="color: #ffa726; font-weight: bold;">flexible circuit structures</span> and <span style="color: #ffa726; font-weight: bold;">aggressive truncation mechanism</span>.
</p>

</div>
<!--

-->

---
layout: center
class: "text-center"
style: "font-size: 3rem; font-weight: bold; color: #4d79ff;"
---

Thank you for listening!

<br>

Any Questions?
