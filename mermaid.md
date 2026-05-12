# LangGraph Architecture Diagram

This file visualizes how the nodes communicate, how the state updates, and which tools are connected to which agents.

If you are using a Markdown previewer that supports Mermaid (like VS Code or GitHub), this block will render as a flowchart!

```mermaid
graph TD
    %% Define styles
    classDef supervisor fill:#f9f,stroke:#333,stroke-width:2px;
    classDef worker fill:#bbf,stroke:#333,stroke-width:2px;
    classDef tools fill:#dfd,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;

    %% Nodes
    START((START))
    END((END))
    
    Supervisor[Supervisor Node\nDecides Next Route]:::supervisor
    
    Researcher[Researcher Agent]:::worker
    Writer[Writer Agent]:::worker
    
    %% Tool Boxes
    subgraph Web Tools
        T1[search_tavily]:::tools
        T2[search_ddg]:::tools
        T3[scrape_webpages]:::tools
    end
    
    subgraph Document Tools
        D1[write_document]:::tools
        D2[read_document]:::tools
        D3[edit_document]:::tools
        D4[create_outline]:::tools
    end

    %% Routing edges
    START --> Supervisor
    
    Supervisor --"Routes to Researcher"--> Researcher
    Supervisor --"Routes to Writer"--> Writer
    Supervisor --"Job Complete"--> END
    
    %% Worker to Tools
    Researcher -. "Calls" .-> T1
    Researcher -. "Calls" .-> T2
    Researcher -. "Calls" .-> T3
    
    Writer -. "Calls" .-> D1
    Writer -. "Calls" .-> D2
    Writer -. "Calls" .-> D3
    Writer -. "Calls" .-> D4
    
    %% Workers return control back to supervisor
    Researcher --"Updates State & Returns"--> Supervisor
    Writer --"Updates State & Returns"--> Supervisor
```

## How the Flow Works
1. Execution starts at **START** and flows immediately to the **Supervisor**.
2. The **Supervisor** reads the `MessagesState` array. It uses its LLM strictly to decide if it should call the Researcher or the Writer.
3. If it calls the **Researcher Agent**, that agent is given a specific internal prompt and uses its **Web Tools** to fetch data.
4. Once the Researcher finishes, it appends its results to the global `MessagesState` array and returns control back to the **Supervisor**.
5. The **Supervisor** reads the new state. It sees the research is done, so it routes to the **Writer Agent**.
6. The Writer Agent uses its **Document Tools** (which are protected by atomic locks) to create files. It appends its final message to the state and returns control.
7. The **Supervisor** sees the final documents are written and routes to **END**, finishing the application.
