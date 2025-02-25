import React from "react";

const OtherTools = () => {
    const tools = [
        "3D-SBDD",
        "Pocket2Mol",
        "GraphBP",
        "TargetDiff",
        "DecompDiff",
        "DiffSBDD",
        "FLAG",
        "DrugGPS",
        "Lingo3DMol",
        "Frag2Seq",
        "LigBuilder",
    ];

    return (
        <div style={{ padding: "20px" }}>
            <h1 style={{ textAlign: "center" }}>Other De Novo based Drug Design Tools</h1>
            <div
                style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
                    gap: "20px",
                    marginTop: "20px",
                }}
            >
                {tools.map((tool, index) => (
                    <div
                        key={index}
                        style={{
                            display: "flex",
                            flexDirection: "column",
                            alignItems: "center",
                            padding: "10px",
                            border: "1px solid #ddd",
                            borderRadius: "8px",
                            boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)",
                            backgroundColor: "#fff",
                        }}
                    >
                        <div
                            style={{
                                width: "80px",
                                height: "80px",
                                backgroundColor: "#f0f0f0",
                                borderRadius: "50%",
                                display: "flex",
                                justifyContent: "center",
                                alignItems: "center",
                                marginBottom: "10px",
                                fontSize: "18px",
                                fontWeight: "bold",
                                color: "#555",
                            }}
                        >
                            {tool.charAt(0)}
                        </div>
                        <p style={{ textAlign: "center", fontSize: "14px", fontWeight: "600" }}>{tool}</p>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default OtherTools;
