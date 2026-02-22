import json
import requests
from typing import List, Dict
from googlesearch import search
from bs4 import BeautifulSoup

from core.memory import Session, get_embedding, cosine_similarity
from core.llm import generate_completion
from database.models import SessionLocal, User, ConceptGraph, DomainSource


def extract_concept_graph(chat_history: List[Dict[str, str]], past_domain: str = None) -> dict:
    """Phase 1: Hidden AI call to extract specific Domain, Nodes, and Edges."""
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    
    past_domain_prompt = ""
    if past_domain:
        past_domain_prompt = f"The user previously studied under the domain '{past_domain}'. If this conversation deeply aligns with that, use it. Otherwise, define a NEW ultra-specific domain.\n"
    
    messages = [
        {"role": "system", "content": "You are a Psychological Profiler and Knowledge Architect analyzing a conversation.\n"
         "Extract the true learning intent into this EXACT JSON structure for a Knowledge Graph:\n"
         "{\n"
         '  "domain": "The ULTRA-SPECIFIC architectural field (e.g., Dark Psychology, NOT General Psychology)",\n'
         '  "core_intent": "What the user ACTUALLY wants to feel, achieve, or understand (e.g., to recognize manipulation tactics in daily life).",\n'
         '  "article_archetype": "The ideal format (e.g., Investigative Journalism, Step-by-Step Tutorial, Comparative Analysis, Academic Paper).",\n'
         '  "exact_phrase_weight": "The most vital 2-4 word exact string they used (e.g., \\"dark side of psychology\\").",\n'
         '  "nodes": [{"id": "topic_name", "status": "known_concept | target_concept | unknown_concept"}],\n'
         '  "edges": [{"source": "node_id_1", "target": "node_id_2", "relationship": "profound creative edge, e.g., \\"is a dangerous application of\\" or \\"is the philosophical opposite of\\" or \\"is a prerequisite for\\" "}]\n'
         "}\n"
         f"{past_domain_prompt}"
         "Ensure all concepts are captured as nodes. Output ONLY pure JSON, no markdown formatting."
        },
        {"role": "user", "content": f"Conversation History:\n{history_text}"}
    ]
    
    response = generate_completion(messages, temperature=0.1)
    try:
        clean_resp = response.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_resp)
    except Exception as e:
        print(f"Error parsing concept graph JSON: {e}")
        return {
            "domain": "General Knowledge", "nodes": [], "edges": []
        }

def evaluate_search_readiness(chat_history: List[Dict[str, str]], turn_count: int) -> bool:
    """The Bouncer LLM: Decides if we stop asking questions and search."""
    if turn_count >= 2:
        return True # Hard cutoff to prevent infinite loops
        
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    messages = [
        {"role": "system", "content": "You are a 'Bouncer' monitoring a conversation. "
         "Does the AI have enough specific context about the user's knowledge baseline to find exactly 3 personalized, targeted articles? "
         "CRITICAL EXCEPTION: If the user explicitly asks you to stop asking questions, demands the articles immediately, or signals extreme impatience, you MUST output YES immediately to bypass the loop. "
         "If yes, output only YES. If no, output only NO."},
        {"role": "user", "content": f"Conversation:\n{history_text}"}
    ]
    response = generate_completion(messages, temperature=0.1)
    return "YES" in response.upper()

def get_verified_sites_for_domain(domain: str, core_intent: str = "", article_archetype: str = "") -> List[str]:
    """Phase 3: Semantic check for domain websites. If new, generate and ping new ones."""
    db = SessionLocal()
    try:
        domain_emb = get_embedding(domain)
        best_match = None
        best_score = -1.0
        
        # 1. Semantic Vector Search
        all_sources = db.query(DomainSource).all()
        for src in all_sources:
            src_emb = src.get_embedding()
            if src_emb:
                score = cosine_similarity(domain_emb, src_emb)
                if score > best_score:
                    best_score = score
                    best_match = src
                    
        if best_match and best_score > 0.85:
            print(f"[Articles Tool] Found semantic domain match: {best_match.domain_name} (Score: {best_score:.2f})")
            return best_match.get_sites()
            
        # 2. Missing Domain Fallback: Generate 5-6 profound sites organically based on intent and archetype.
        print(f"[Articles Tool] New Domain detected ({domain}). Generating organic sources...")
        messages = [
            {"role": "system", "content": "You are an expert librarian and domain sniper. "
             f"The user wants to explore the domain '{domain}'. "
             f"Their core intent is: '{core_intent}'. The ideal reading format is: '{article_archetype}'. "
             "Give me exactly 6 of the absolute best, most profoundly respected, high-quality website domain names that match this exact vibe. "
             "If the topic requires human subjectivity or niche hacks, include specific platforms like 'reddit.com' or 'medium.com'. "
             "Return ONLY a pure JSON array of raw domain strings (e.g., ['psychologytoday.com', 'apa.org']). No markdown."},
            {"role": "user", "content": f"Subject: {domain}"}
        ]
        response = generate_completion(messages, temperature=0.2)
        
        try:
            clean_resp = response.replace("```json", "").replace("```", "").strip()
            proposed_sites = json.loads(clean_resp)
        except:
            proposed_sites = ["medium.com", "wikipedia.org", "reddit.com"] # Safest ultimate fallback
            
        verified_sites = []
        for site in proposed_sites:
            # Ping test to verify the domain isn't dead
            url = f"https://{site}" if not site.startswith("http") else site
            real_domain = site.replace("https://", "").replace("http://", "").split("/")[0]
            try:
                # Add headers to avoid basic bot blocks
                headers = {'User-Agent': 'Mozilla/5.0'}
                res = requests.get(url, headers=headers, timeout=3)
                if res.status_code == 200:
                    verified_sites.append(real_domain)
            except Exception as e:
                print(f"[Ping Failed] {site}: {e}")
                
        # Ensure we have at least something fallback if all fail
        if not verified_sites:
            verified_sites = ["medium.com", "wikipedia.org"]
            
        # 3. Save new domain and verified sites so the system gets permanently smarter
        new_source = DomainSource(domain_name=domain)
        new_source.set_embedding(domain_emb)
        new_source.set_sites(verified_sites)
        db.add(new_source)
        db.commit()
        
        return verified_sites
        
    finally:
        db.close()

def generate_queries_from_graph(graph: Dict, verified_sites: List[str], core_intent: str = "", article_archetype: str = "") -> List[str]:
    """Phase 3 (Sniper): Select the single best domain and generate exact-match queries."""
    
    # SNR (Sniper Re-Ranking): Pick the single best domain for the exact vibe.
    sniper_messages = [
        {"role": "system", "content": f"You are a routing Sniper. Given the user's core intent: '{core_intent}' and desired format: '{article_archetype}', pick the SINGLE BEST domain from the provided list to search. If none are great fits, default to 'medium.com' or 'wikipedia.org'. Return ONLY the raw domain string and nothing else."},
        {"role": "user", "content": f"Options: {', '.join(verified_sites)}"}
    ]
    sniper_domain = generate_completion(sniper_messages, temperature=0.1).strip()
    
    # Fallback if the LLM hallucinated
    if sniper_domain not in verified_sites and "medium.com" not in sniper_domain and "wikipedia.org" not in sniper_domain:
         sniper_domain = verified_sites[0]
         
    sites_filter = f"site:{sniper_domain}"
    print(f"[Articles Tool] Sniper Strategy selected primary domain: {sniper_domain}")
    
    exact_phrase = graph.get("exact_phrase_weight", "")
    
    # Extract target and unknown concepts from the nodes array
    unknowns = []
    for node in graph.get("nodes", []):
        if node.get("status") in ["target_concept", "unknown_concept"]:
            unknowns.append(node.get("id"))
            
    unknowns_str = ", ".join(unknowns) if unknowns else "general tutorial"
    
    messages = [
        {"role": "system", "content": "You generate specific DuckDuckGo search queries. "
         f"Given these unknown target topics the user needs to learn: {unknowns_str}, return exactly 3 highly targeted search strings. "
         f"If possible, incorporate the vital exact string: \"{exact_phrase}\" in quotes. "
         f"You MUST append '{sites_filter}' to the end of EVERY search string. "
         "Return ONLY a pure JSON array of strings. No markdown."},
        {"role": "user", "content": json.dumps(graph)}
    ]
    response = generate_completion(messages, temperature=0.1)
    try:
        clean_resp = response.replace("```json", "").replace("```", "").strip()
        queries = json.loads(clean_resp)
        return queries[:3]
    except Exception as e:
        print(f"Error parsing queries JSON: {e}")
        return [f"\"{exact_phrase}\" {unknowns_str} {sites_filter}"]

def execute_searches(queries: List[str]) -> List[Dict]:
    """Fetch top 3 results per query via Google Search and organically scrape the content."""
    raw_results = []
    
    for q in queries:
        try:
            print(f"[Scraper] Executing Google Search for: {q}")
            # sleep_interval introduces a forced delay between requests to bypass rate-limits natively
            for url in search(q, num_results=3, sleep_interval=2):
                try:
                    # `googlesearch` only returns the URL. We must organically fetch the page snippet.
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                        'Accept-Language': 'en-US,en;q=0.5'
                    }
                    res = requests.get(url, headers=headers, timeout=5)
                    if res.status_code == 200:
                        soup = BeautifulSoup(res.text, 'html.parser')
                        title = soup.title.string if soup.title else "Untitled Article"
                        
                        # Remove unwanted tags
                        for script in soup(["script", "style", "nav", "header", "footer"]):
                            script.decompose()
                            
                        # Extract raw text from the remaining body
                        text = soup.get_text(separator=' ', strip=True)
                        
                        raw_results.append({
                            "query": q,
                            "title": title.strip(),
                            "href": url,
                            "snippet": text[:1000] # Cap snippet length for the embedding model
                        })
                    else:
                        print(f"[Scraper] Failed to fetch {url}: HTTP {res.status_code}")
                except Exception as e:
                    print(f"[Scraper] Failed to extract snippet from {url}: {str(e)[:50]}")
                    continue
        except Exception as search_e:
            err_str = str(search_e)
            if "429" in err_str:
                print(f"[Scraper] Error 429 (Too Many Requests) hit on Google Search. Halting fetching to prevent IP ban.")
                break # Stop searching and rely on the results gathered (or fallback)
            print(f"[Scraper] Google Search failed for '{q}': {err_str[:50]}")
                
    return raw_results

def filter_and_summarize(results: List[Dict], graph: Dict) -> str:
    """Phase 4 (Re-Ranking): mathematically sorts snippets by core intent before summarizing."""
        
    core_intent = graph.get("core_intent", "learning the basics")
    print(f"[Articles Tool] Semantically re-ranking {len(results)} results against intent: '{core_intent}'")
    
    intent_emb = get_embedding(core_intent)
    
    # Calculate similarity score for each snippet
    for r in results:
        snippet_emb = get_embedding(r['snippet'])
        r['score'] = cosine_similarity(intent_emb, snippet_emb)
        
    # Sort by score descending and take the absolute top 3
    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    top_results = results[:3]
        
    messages = [
        {"role": "system", "content": "You are a curator of profound knowledge. "
         f"The user's core intent is: '{core_intent}'. "
         "Take these mathematically pre-filtered articles and write a brilliant 2-3 sentence summary for each, explaining exactly how it fulfills their intent. "
         "If the search results seem completely unrelated or broke due to rate limits, use the Graph to explain the core concepts directly, acting as the article itself. "
         "Format the output EXACTLY like this for each article block:\n\n"
         "Link: [The URL]\n"
         "Summary: [Exactly 2 lines explaining why this is useful for their intent.]\n\n"
         "CRITICAL RULES:\n"
         "1. ALWAYS include the Link first. NEVER omit the URL. If the URL is missing, use the search query to explain the concept yourself.\n"
         "2. The summary MUST be exactly two sentences, no longer.\n"
         "3. Separate each block with two blank lines. Do not use bolding, asterisks, or emojis."},
        {"role": "user", "content": f"User Graph: {json.dumps(graph)}\n\nSearch Results: {json.dumps(top_results)}"}
    ]
    
    return generate_completion(messages, temperature=0.3)


def handle_articles_tool(session: Session, message: str) -> str:
    """
    The conversational entry point for the /articles tool.
    Phase 2: Plays "The Playful Mentor" until the Bouncer LLM intercepts.
    """
    session.add_message("user", message)
    
    if message.lower() == "done":
        session.pop_tool()
        return "You have exited the Articles Tool. You are back in standard chat."

    # Analyze chat turns (divide by 2 since user+assistant is a turn Pair)
    # The initial trigger is 1 user message, so turns = len // 2.
    tool_messages = len(session.chat_history) - session.tool_start_idx
    turn_count = tool_messages // 2 
    print(f"[Articles Tool] Evaluating Search Readiness (Turn {turn_count})...")
    
    # Bouncer LLM decides if we should stop playing and start searching
    is_ready = evaluate_search_readiness(session.chat_history, turn_count)
    
    if not is_ready:
        print("[Articles Tool] Not ready. Invoking Playful Teacher persona...")
        # Pass any historical domain from RAG so the extraction knows what to map to
        past_domain = None
        if hasattr(session, 'active_rag_context') and session.active_rag_context:
            import re
            match = re.search(r"Domain: (.*?)\n", session.active_rag_context)
            if match: past_domain = match.group(1)
            
        current_graph = extract_concept_graph(session.chat_history, past_domain=past_domain)
        
        target_nodes = [n.get('id') for n in current_graph.get('nodes', []) if n.get('status') == 'target_concept']
        targets = ", ".join(target_nodes) if target_nodes else "this new topic"
        
        edges = [f"{e.get('source')} {e.get('relationship')} {e.get('target')}" for e in current_graph.get('edges', [])]
        edges_str = ", ".join(edges) if edges else "None"
        
        core_intent = current_graph.get('core_intent', 'Learn the basics')
        
        system_prompt = (
            "You are Morarc, a sharp, witty, and slightly provocative 'Socratic Challenger'. "
            "Your job is to figure out the user's true underlying intent so you can find them the perfect articles. "
            f"The user wants to explore: {targets}. Their initially detected intent is: '{core_intent}'. "
            f"You have mapped these conceptual relationships so far: {edges_str}. "
            "Do NOT act like a generic AI tutor or teacher. Do NOT ask academic or formal questions. "
            "Instead, play devil's advocate. Challenge their assumptions. Ask ONE sharp, provocative, and highly conversational question that forces them to defend exactly what angle they are looking for. "
            "Keep it incredibly brief, punchy, and sound like a highly intelligent human texting a friend. "
            "CRITICAL INSTRUCTION: You are strictly forbidden from using any emojis whatsoever. "
            "If asked about what tools you have, explicitly state that you have one primary tool: `/articles <topic>`, which performs deep semantic web searches, and that you are currently using it."
        )
        
        # Inject RAG context if we recognized their past learning history
        if session.active_rag_context:
            system_prompt += f"\n\n{session.active_rag_context}"
            
        messages = [{"role": "system", "content": system_prompt}] + session.chat_history
        reply = generate_completion(messages, temperature=0.7) # Slightly higher temp for personality
        
        session.add_message("assistant", reply)
        return reply

    # 3. Bouncer triggered! Finalize Graph, Save & Search
    print("[Articles Tool] Ready threshold met! Finalizing Knowledge Graph...")
    
    # Check if we have a historical domain to map to
    past_domain = None
    if hasattr(session, 'active_rag_context') and session.active_rag_context:
        import re
        match = re.search(r"Domain: (.*?)\n", session.active_rag_context)
        if match: past_domain = match.group(1)
        
    final_graph = extract_concept_graph(session.chat_history, past_domain=past_domain)
    domain = final_graph.get("domain", "General Knowledge")
    
    # Phase 4 Update: Unified Graph Merging & Compression
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.phone_number == session.phone_number).first()
        if user:
            existing_graph = None
            if hasattr(session, 'active_graph_id') and session.active_graph_id:
                existing_graph = db.query(ConceptGraph).filter(ConceptGraph.id == session.active_graph_id).first()
                
            if existing_graph and existing_graph.domain == domain:
                # Merge logic: Ask LLM to append/overwrite old graph with new conversation
                print(f"[Articles Tool] Merging into existing domain graph: {domain}")
                old_data = existing_graph.get_graph_data()
                
                merge_msgs = [
                    {"role": "system", "content": "You merge knowledge graphs. You will receive an OLD JSON Graph, and a NEW JSON Graph representing a recent conversation. "
                     "Your job is to merge them into a single final JSON. "
                     "RULES:\n"
                     "1. Append any new nodes or edges.\n"
                     "2. If an old node appears in the new graph with an updated status (e.g. they learned it), OVERWRITE the old status.\n"
                     "3. Output ONLY pure JSON matching the standard format (`domain`, `nodes`, `edges`)."},
                    {"role": "user", "content": f"OLD:\n{json.dumps(old_data)}\n\nNEW:\n{json.dumps(final_graph)}"}
                ]
                merged_resp = generate_completion(merge_msgs, temperature=0.1)
                try:
                    merged_json = json.loads(merged_resp.replace("```json", "").replace("```", "").strip())
                    final_graph = merged_json # We use the merged graph for the search execution too!
                    
                    # Compression Logic: If > 20 nodes, compress by 40%
                    if len(final_graph.get("nodes", [])) > 20:
                        print(f"[Articles Tool] Graph > 20 nodes. Executing 40% LLM Compression...")
                        comp_msgs = [
                             {"role": "system", "content": "This Knowledge Graph is too large. Compress the node count by at least 40%. "
                              "Combine highly related small nodes into single foundational nodes, and delete outdated edges that are no longer strictly necessary to understand the core map. "
                              "Output ONLY pure JSON. Keep the `domain`, `nodes`, and `edges` format."},
                             {"role": "user", "content": json.dumps(final_graph)}
                        ]
                        comp_resp = generate_completion(comp_msgs, temperature=0.1)
                        final_graph = json.loads(comp_resp.replace("```json", "").replace("```", "").strip())

                    existing_graph.set_graph_data(final_graph)
                    existing_graph.set_embedding(get_embedding(domain))
                    db.commit()
                except Exception as eval_e:
                    print(f"Error during Graph Merge/Compression: {eval_e}")
            else:
                # Create brand new standalone domain graph row
                print(f"[Articles Tool] Creating new domain graph: {domain}")
                new_graph = ConceptGraph(
                    user_id=user.id,
                    domain=domain,
                )
                new_graph.set_graph_data(final_graph)
                new_graph.set_embedding(get_embedding(domain))
                db.add(new_graph)
                db.commit()
    except Exception as e:
        print(f"Error saving concept graph to Database: {e}")
    finally:
        db.close()

    # Search Pipeline (Phase 3)
    try:
        core_int = final_graph.get("core_intent", "")
        arc_arch = final_graph.get("article_archetype", "")
        
        verified_sites = get_verified_sites_for_domain(domain, core_int, arc_arch)
        queries = generate_queries_from_graph(final_graph, verified_sites, core_int, arc_arch)
        raw_results = execute_searches(queries)
        final_output = filter_and_summarize(raw_results, final_graph)
        
        # Format the final response
        response = f"Your interested concept graph fully evoked.\n\n"
        response += final_output
        response += "\n\n_(Reply with '/stop' to end chat or 'done' to exit tool)_"
        
        # Clean up the tool state since the goal was successfully achieved
        session.pop_tool()
        return response
        
    except Exception as e:
        session.pop_tool()
        error_msg = f"An error occurred while generating articles: {str(e)}"
        print(error_msg)
        return error_msg
