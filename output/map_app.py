import streamlit as st
import folium
from streamlit_folium import st_folium

# --- Configurações iniciais ---
st.set_page_config(layout="wide")
st.title("🗺️ Editor Interativo de Múltiplas Rotas")

# Centro e zoom iniciais do mapa
BASE_CENTER = [-15.7684, -47.8696]
BASE_ZOOM = 14

# Inicializar estado para múltiplas rotas
if "routes" not in st.session_state:
    st.session_state.routes = []
if "current_pts" not in st.session_state:
    st.session_state.current_pts = []
if "current_color" not in st.session_state:
    st.session_state.current_color = "#FF6600"

# --- Função para recriar o mapa com todas as rotas ---
def criar_mapa():
    m = folium.Map(location=BASE_CENTER, zoom_start=BASE_ZOOM)

    # Desenhar rotas salvas
    for idx, route in enumerate(st.session_state.routes, start=1):
        folium.PolyLine(
            route['pts'],
            color=route['color'],
            weight=4,
            opacity=0.8,
            tooltip=f"Rota {idx}"
        ).add_to(m)
        for lat, lng in route['pts']:
            folium.CircleMarker((lat, lng), radius=3, color=route['color'], fill=True).add_to(m)

    # Desenhar rota em edição
    if st.session_state.current_pts:
        folium.PolyLine(
            st.session_state.current_pts,
            color=st.session_state.current_color,
            weight=4,
            opacity=0.8,
            dash_array="5,5",
            tooltip="Rota Atual"
        ).add_to(m)
        for lat, lng in st.session_state.current_pts:
            folium.CircleMarker((lat, lng), radius=4, color=st.session_state.current_color).add_to(m)
    return m

# --- Sidebar de controle ---
with st.sidebar:
    st.header("Controles de Rotas")
    st.markdown("**Clique no mapa** para adicionar nós à rota atual.")

    # Escolha de cor da rota em edição
    st.session_state.current_color = st.color_picker("Cor da nova rota", st.session_state.current_color)

    # Botão para remover último nó adicionado
    if st.button("↩️ Remover último nó"):
        if st.session_state.current_pts:
            st.session_state.current_pts.pop()
            st.info("Último nó removido.")
        else:
            st.warning("Nenhum nó para remover.")

    # Botão para salvar rota atual como rota definitiva
    if st.button("➕ Salvar Rota Atual"):
        if len(st.session_state.current_pts) >= 2:
            st.session_state.routes.append({
                'pts': st.session_state.current_pts.copy(),
                'color': st.session_state.current_color
            })
            st.session_state.current_pts = []
            st.success("Rota adicionada! Defina nova cor ou comece outra rota.")
        else:
            st.error("Insira pelo menos 2 pontos antes de salvar a rota.")

    # Botão para limpar completamente a rota em edição
    if st.button("🗑️ Limpar Rota Atual"):
        st.session_state.current_pts = []
        st.info("Rota atual limpa.")

    # Botão para gerar HTML com todas as rotas
    if st.button("💾 Gerar HTML com Rotas"):
        m_final = criar_mapa()
        html_str = m_final.get_root().render()
        filename = "grafo_mult_rotas.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_str)
        st.success(f"Arquivo salvo: {filename}")

# --- Exibição do mapa e captura de cliques ---
m = criar_mapa()
map_data = st_folium(m, width=900, height=600, returned_objects=["last_clicked"])

# Adicionar ponto se clicado
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lng = map_data["last_clicked"]["lng"]
    if not st.session_state.current_pts or st.session_state.current_pts[-1] != (lat, lng):
        st.session_state.current_pts.append((lat, lng))

# Exibir nós da rota atual
st.markdown("**Nós da rota atual:**")
if st.session_state.current_pts:
    for i, (lat, lng) in enumerate(st.session_state.current_pts, 1):
        st.write(f"{i}. Lat: {lat:.6f}, Lng: {lng:.6f}")
else:
    st.write("Nenhum ponto definido.")

# Exibir resumo das rotas salvas
st.markdown("## Rotas Salvas")
if st.session_state.routes:
    for idx, route in enumerate(st.session_state.routes, start=1):
        st.write(f"- Rota {idx}: {len(route['pts'])} pontos, cor {route['color']}")
else:
    st.write("Nenhuma rota salva ainda.")
