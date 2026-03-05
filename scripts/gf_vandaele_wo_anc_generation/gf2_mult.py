from typing import List, Tuple
import sys


from typing import Dict, List, Union
import os

def create_qubit_mapping(num_qubits: int, 
                         label_format: str = "{}{}") -> Dict[int, str]:
    """
    Create a mapping from qubit indices to labels.
    Similar to Rust example: a0, b0, c0, d0, a1, b1, ...
    """
    prefixes = ["a", "b", "c", "d"]
    mapping = {}
    
    qubits_per_prefix = num_qubits // len(prefixes)
    remainder = num_qubits % len(prefixes)
    
    idx = 0
    for prefix in prefixes:
        for j in range(qubits_per_prefix):  # +1 to handle remainder
            if idx >= num_qubits:
                break
            mapping[idx] = label_format.format(prefix, j)
            idx += 1
    
    return mapping

def circuit_to_qc(circuit: QuantumCircuit, 
                  filename: str,
                  qubit_label_format: str = "{}{}") -> None:
    """
    Convert a Qiskit circuit to .qc format with specific gates.
    
    Args:
        circuit: Qiskit QuantumCircuit object
        filename: Output .qc filename
        qubit_label_format: Format string for qubit labels (default: "{}{}")
    
    Supported gates: H, CX, CCX, CCZ
    """
    
    # Check if circuit contains only supported gates
    supported_gates = {'h', 'cx', 'ccx', 'ccz'}
    
    for instruction in circuit.data:
        gate_name = instruction.operation.name.lower()
        if gate_name not in supported_gates:
            raise ValueError(f"Unsupported gate: {gate_name}. "
                           f"Only {supported_gates} are supported.")
    
    # Create mapping of qubit indices to labels
    qubit_map = create_qubit_mapping(circuit.num_qubits, qubit_label_format)
    
    # Write to .qc file
    with open(filename, 'w') as f:
        # Write header with variable names
        f.write(".v")
        for i in range(circuit.num_qubits):
            f.write(f" {qubit_map[i]}")
        f.write("\n")
        
        # Write input qubits (first half by default)
        f.write(".i")
        num_inputs = circuit.num_qubits // 2
        for i in range(num_inputs):
            f.write(f" {qubit_map[i]}")
        f.write("\n")
        
        # Begin circuit
        f.write("BEGIN\n")
        
        # Write gates
        for instruction in circuit.data:
            gate_name = instruction.operation.name.lower()
            qubits = [qubit._index for qubit in instruction.qubits]
            
            # Map gate names to .qc format
            if gate_name == 'h':
                f.write(f"H {qubit_map[qubits[0]]}\n")
            elif gate_name == 'cx':
                f.write(f"tof {qubit_map[qubits[0]]} {qubit_map[qubits[1]]}\n")
            elif gate_name == 'ccx':
                f.write(f"tof {qubit_map[qubits[0]]} {qubit_map[qubits[1]]} {qubit_map[qubits[2]]}\n")
            elif gate_name == 'ccz':
                f.write(f"Z {qubit_map[qubits[0]]} {qubit_map[qubits[1]]} {qubit_map[qubits[2]]}\n")
        
        f.write("END\n")
    
    print(f"Circuit saved to {filename}")

def reduction_cnot_circuit(p: List[int]) -> QuantumCircuit:
    """Create a reduction CNOT circuit based on polynomial p"""
    n = len(p)
    offset = 3 * n
    circ = QuantumCircuit(4 * n, name="reduction_cnot")
    
    # First loop
    for i in range(1, n - 1):
        for j in range(n - i, n):
            if p[j] == 1:
                circ.cx(offset + j - n + i, offset + i)
    
    # Second loop
    for i in range(n - 2, -1, -1):
        for j in range(1, n - i):
            if p[j] == 1:
                circ.cx(offset + i + j, offset + i)
    
    return circ

def gf_mult_synth_rec(circ: QuantumCircuit, a: List[int], b: List[int], 
                      c: List[int], d: List[int]):
    """Recursive function for GF multiplication synthesis"""
    a = a.copy()
    b = b.copy()
    c = c.copy()
    d = d.copy()
    if len(a) == 1:
        # CCZ gate decomposition into Toffoli and T gates
        # Using standard CCZ decomposition: CCZ = H(2) • CCX • H(2)
        print(len(a), len(b), len(c))
        if a[0] == -1 or b[0] == -1 or c[0] == -1:
            return
        target = c[0]
        circ.ccz(a[0], b[0], target)
        return
    
    if len(a) % 2 == 1:
        a.append(-1)  # Placeholder
        b.append(-1)  # Placeholder
        c.append(d.pop(0))
        d.append(-1)  # Placeholder
        d.append(-1)  # Placeholder
    
    mid = len(a) // 2
    a_l, a_r = a[:mid], a[mid:]
    b_l, b_r = b[:mid], b[mid:]
    c_l, c_r = c[:mid], c[mid:]
    d_l, d_r = d[:mid], d[mid:]
    
    # First set of CNOTs
    for i in range(mid):
        if a_r[i] != -1 and a_l[i] != -1:
            circ.cx(a_r[i], a_l[i])
        if b_r[i] != -1 and b_l[i] != -1:
            circ.cx(b_r[i], b_l[i])
    
    # Recursive call
    gf_mult_synth_rec(circ, a_l, b_l, c_r, d_l)
    
    # Reverse first set of CNOTs
    for i in range(mid):
        if b_r[i] != -1 and b_l[i] != -1:
            circ.cx(b_r[i], b_l[i])
        if a_r[i] != -1 and a_l[i] != -1:
            circ.cx(a_r[i], a_l[i])
    
    # Second set of CNOTs
    for i in range(mid):
        if c_r[i] != -1 and c_l[i] != -1:
            circ.cx(c_r[i], c_l[i])
        if d_l[i] != -1 and c_r[i] != -1:
            circ.cx(d_l[i], c_r[i])
        if d_r[i] != -1 and d_l[i] != -1:
            circ.cx(d_r[i], d_l[i])
    
    # Recursive calls
    gf_mult_synth_rec(circ, a_r, b_r, c_r, d_l)
    gf_mult_synth_rec(circ, a_l, b_l, c_l, c_r)
    
    # Reverse second set of CNOTs
    for i in range(mid):
        if d_r[i] != -1 and d_l[i] != -1:
            circ.cx(d_r[i], d_l[i])
        if d_l[i] != -1 and c_r[i] != -1:
            circ.cx(d_l[i], c_r[i])
        if c_r[i] != -1 and c_l[i] != -1:
            circ.cx(c_r[i], c_l[i])

def gf_mult_synth(p: List[int]) -> QuantumCircuit:
    """Synthesize GF(2^n) multiplication circuit"""
    n = len(p)
    circ = QuantumCircuit(4 * n, name=f"gf2^{n}_mult")
    
    # Apply H gates to qubits in range [2n, 3n)
    for i in range(2 * n, 3 * n):
        circ.h(i)
    
    # Apply CNOTs from [2n, 3n) to [3n, 4n)
    for i in range(2 * n, 3 * n):
        circ.cx(i, i + n)
    
    # Create and append reduction CNOT circuit
    cnot_circ = reduction_cnot_circuit(p)
    
    # Append reversed reduction CNOT circuit
    reversed_circ = QuantumCircuit(4 * n)
    for instruction in reversed(cnot_circ.data):
        reversed_circ.append(instruction.operation, instruction.qubits)
    
    circ = circ.compose(reversed_circ)
    
    # Recursive synthesis
    a = list(range(n))
    b = list(range(n, 2 * n))
    c = list(range(2 * n, 3 * n))
    d = list(range(3 * n, 4 * n))
    
    gf_mult_synth_rec(circ, a, b, c, d)
    
    # Append forward reduction CNOT circuit
    circ = circ.compose(cnot_circ)
    
    # Reverse CNOTs from [2n, 3n) to [3n, 4n)
    for i in range(3 * n - 1, 2 * n - 1, -1):
        circ.cx(i, i + n)
    
    # Apply H gates to qubits in range [2n, 3n)
    for i in range(2 * n, 3 * n):
        circ.h(i)
    
    return circ

def main():
    """Main function similar to Rust version"""
    # Get indices from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python gf_mult.py <index1> <index2> ...")
        sys.exit(1)
    
    try:
        indices = [int(arg) for arg in sys.argv[1:]]
    except ValueError:
        print("Error: All arguments must be integers")
        sys.exit(1)
    
    # Create polynomial representation
    if indices:
        max_index = max(indices)
        p = [0] * (max_index)
        for index in indices:
            if index < max_index:
                p[index] = 1
    else:
        print("No indices provided.")
        return
    
    n = len(p)
    print(f"Creating GF(2^{n}) multiplication circuit")
    print(f"Polynomial representation: {p}")
    
    # Generate circuit
    circ: QuantumCircuit = gf_mult_synth(p)
    
    # Print circuit info
    print(f"\nCircuit created:")
    print(f"  Qubits: {circ.num_qubits}")
    print(f"  Depth: {circ.depth()}")
    print(f"  Gate count: {circ.size()}")
    
    # Save to file (QASM format)
    filename = f"circuits/gf2^{n}_mult.qasm"
    # with open('circuit.qasm', 'w') as stream:
        # dump(circ, stream)
    circuit_to_qc(circ, "circuit.qc")
    # circ.qas(filename=filename)
    print(f"\nCircuit saved to: {filename}")
  
    # Also print a simplified version
    print("\nSimplified circuit diagram:")
    # print(circ.draw(output='text', fold=-1))

if __name__ == "__main__":
    main()