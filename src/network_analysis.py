"""
Network Analysis Module - Phase 8C
Builds provider-beneficiary-physician graphs using NetworkX.
Only uses reliable identifiers.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAW_DIR = Path("data/raw")
REPORT_DIR = Path("reports")

def build_provider_network(sample_size: int = None) -> nx.Graph:
    """
    Build graph with providers, beneficiaries, physicians as nodes.
    Edges: provider-beneficiary, provider-physician via claims.
    """
    def find(pattern): 
        m = sorted(RAW_DIR.glob(pattern))
        return m[0] if m else None
    
    inpatient_path = find("Train_Inpatientdata-*.csv")
    outpatient_path = find("Train_Outpatientdata-*.csv")
    
    claims = []
    for path in [inpatient_path, outpatient_path]:
        if path:
            usecols = ["Provider","BeneID","AttendingPhysician","OperatingPhysician","ClaimID"]
            # Load only needed cols to save memory
            try:
                df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
            except:
                df = pd.read_csv(path, low_memory=False)
                df = df[[c for c in df.columns if c in usecols]]
            claims.append(df)
    if not claims:
        return nx.Graph()
    claims_df = pd.concat(claims, ignore_index=True)
    # Sample if needed
    if sample_size and len(claims_df) > sample_size:
        claims_df = claims_df.sample(n=sample_size, random_state=42)
    
    G = nx.Graph()
    
    # Add provider-beneficiary edges
    # Create bipartite edges
    for _, row in claims_df.iterrows():
        provider = row["Provider"]
        bene = row["BeneID"]
        if pd.notna(provider) and pd.notna(bene):
            G.add_node(provider, type="provider")
            G.add_node(bene, type="beneficiary")
            G.add_edge(provider, bene, relation="treated")
        
        # Provider-physician
        for phys_col in ["AttendingPhysician","OperatingPhysician"]:
            if phys_col in row and pd.notna(row[phys_col]) and str(row[phys_col]).strip() not in ["", "nan"]:
                phys = str(row[phys_col]).strip()
                G.add_node(phys, type="physician")
                if pd.notna(provider):
                    G.add_edge(provider, phys, relation="employs")
    
    print(f"Network built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Providers: {sum(1 for n,d in G.nodes(data=True) if d.get('type')=='provider')}")
    print(f"  Beneficiaries: {sum(1 for n,d in G.nodes(data=True) if d.get('type')=='beneficiary')}")
    print(f"  Physicians: {sum(1 for n,d in G.nodes(data=True) if d.get('type')=='physician')}")
    return G

def analyze_network(G: nx.Graph) -> dict:
    """Compute centrality, connected components, suspicious clusters."""
    if G.number_of_nodes() == 0:
        return {"error": "Empty graph"}
    
    # Connected components
    components = list(nx.connected_components(G))
    comp_sizes = [len(c) for c in components]
    
    # Degree centrality
    deg_cent = nx.degree_centrality(G)
    # Top central nodes
    top_central = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Provider centrality only
    provider_nodes = [n for n,d in G.nodes(data=True) if d.get('type')=='provider']
    provider_deg = {n: deg_cent[n] for n in provider_nodes if n in deg_cent}
    top_providers = sorted(provider_deg.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Shared beneficiary analysis: providers sharing > X beneficiaries
    # Build provider-beneficiary bipartite projection
    # For each beneficiary, list providers
    bene_to_providers = {}
    for u,v in G.edges():
        # Identify types
        ut = G.nodes[u].get('type')
        vt = G.nodes[v].get('type')
        if (ut=='provider' and vt=='beneficiary'):
            bene_to_providers.setdefault(v, []).append(u)
        elif (vt=='provider' and ut=='beneficiary'):
            bene_to_providers.setdefault(u, []).append(v)
    
    # Count shared beneficiaries between provider pairs
    from collections import Counter
    shared_counts = Counter()
    for bene, provs in bene_to_providers.items():
        if len(provs) > 1:
            # All pairs
            for i in range(len(provs)):
                for j in range(i+1, len(provs)):
                    pair = tuple(sorted([provs[i], provs[j]]))
                    shared_counts[pair] += 1
    
    # Flag unusually high shared beneficiary pairs (>5 shared)
    suspicious_shared = {k:v for k,v in shared_counts.items() if v > 5}
    
    # Shared physician
    phys_to_providers = {}
    for u,v in G.edges():
        ut = G.nodes[u].get('type')
        vt = G.nodes[v].get('type')
        if (ut=='provider' and vt=='physician'):
            phys_to_providers.setdefault(v, []).append(u)
        elif (vt=='provider' and ut=='physician'):
            phys_to_providers.setdefault(u, []).append(v)
    
    shared_phys_counts = Counter()
    for phys, provs in phys_to_providers.items():
        if len(provs) > 1:
            for i in range(len(provs)):
                for j in range(i+1, len(provs)):
                    pair = tuple(sorted([provs[i], provs[j]]))
                    shared_phys_counts[pair] += 1
    suspicious_phys_shared = {k:v for k,v in shared_phys_counts.items() if v > 2}
    
    analysis = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_components": len(components),
        "largest_component_size": max(comp_sizes) if comp_sizes else 0,
        "avg_component_size": np.mean(comp_sizes) if comp_sizes else 0,
        "top_central_nodes": top_central,
        "top_providers_by_centrality": top_providers,
        "providers_with_high_shared_beneficiaries": suspicious_shared,
        "providers_with_shared_physicians": suspicious_phys_shared,
        "disclaimer": "Shared connections indicate co-occurrence, not proof of collusion. Requires reliable identifiers and human review."
    }
    
    # Save reports
    REPORT_DIR.mkdir(exist_ok=True)
    # Save suspicious pairs to CSV
    if suspicious_shared:
        shared_df = pd.DataFrame([{"Provider1":k[0],"Provider2":k[1],"SharedBeneficiaries":v} for k,v in suspicious_shared.items()])
        shared_df = shared_df.sort_values("SharedBeneficiaries", ascending=False)
        shared_df.to_csv(REPORT_DIR / "network_shared_beneficiaries.csv", index=False)
    else:
        pd.DataFrame(columns=["Provider1","Provider2","SharedBeneficiaries"]).to_csv(REPORT_DIR / "network_shared_beneficiaries.csv", index=False)
    
    if suspicious_phys_shared:
        phys_df = pd.DataFrame([{"Provider1":k[0],"Provider2":k[1],"SharedPhysicians":v} for k,v in suspicious_phys_shared.items()])
        phys_df.to_csv(REPORT_DIR / "network_shared_physicians.csv", index=False)
    else:
        pd.DataFrame(columns=["Provider1","Provider2","SharedPhysicians"]).to_csv(REPORT_DIR / "network_shared_physicians.csv", index=False)
    
    # Save centrality
    centrality_df = pd.DataFrame([{"Node":k,"Centrality":v,"Type":G.nodes[k].get('type')} for k,v in deg_cent.items()])
    centrality_df = centrality_df.sort_values("Centrality", ascending=False).head(100)
    centrality_df.to_csv(REPORT_DIR / "network_centrality.csv", index=False)
    
    print(f"Network analysis: {len(components)} components, {len(suspicious_shared)} suspicious shared bene pairs, {len(suspicious_phys_shared)} shared phys pairs")
    return analysis

def plot_network(G: nx.Graph, save_path: Path = REPORT_DIR / "network_graph.png", sample_nodes: int = 300):
    """Plot sampled network (full may be too large)."""
    if G.number_of_nodes() == 0:
        return
    # Sample largest component
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if components:
        largest = components[0]
        # Sample from largest
        if len(largest) > sample_nodes:
            sampled_nodes = list(largest)[:sample_nodes]
            subG = G.subgraph(sampled_nodes)
        else:
            subG = G.subgraph(largest)
        
        plt.figure(figsize=(12,12))
        pos = nx.spring_layout(subG, k=0.15, iterations=20, seed=42)
        # Color by type
        color_map = {"provider":"red","beneficiary":"lightblue","physician":"green"}
        colors = [color_map.get(G.nodes[n].get('type'), 'gray') for n in subG.nodes()]
        nx.draw(subG, pos, node_color=colors, node_size=50, with_labels=False, edge_color='gray', alpha=0.5, width=0.5)
        # Legend
        for t,c in color_map.items():
            plt.scatter([],[], c=c, label=t)
        plt.legend()
        plt.title(f"Provider Network (sample {len(subG.nodes())} nodes from largest component)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved network plot to {save_path}")

if __name__ == "__main__":
    G = build_provider_network(sample_size=20000)
    analysis = analyze_network(G)
    print(analysis)
    plot_network(G)
