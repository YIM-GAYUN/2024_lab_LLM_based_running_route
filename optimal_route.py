import folium
import osmnx as ox
import networkx as nx
import googlemaps
from osmnx import distance as ox_distance

# Google Maps API 키 설정
gmaps = googlemaps.Client(key='본인 api key')

# 사용자 선호도 설정 (출발지: 이화여대 정문, 도착지: 서울역)
user_preference = {
    "start": "이화여자대학교, 서울특별시 서대문구 대현동",
    "end": "서울역, 서울특별시 중구 봉래동2가",
    "incline": "low",
    "environment": ["공원", "강", "park", "river"],
    "facilities": ["화장실", "주차장", "toilet", "parking"]
}

# 출발지와 목적지 좌표 가져오기
start_geocode = gmaps.geocode(user_preference['start'], language='ko')
end_geocode = gmaps.geocode(user_preference['end'], language='ko')

# 주소 유효성 검사 및 좌표 설정
if not start_geocode or not end_geocode:
    print("Error: One or both of the addresses could not be found.")
else:
    start_location = start_geocode[0]['geometry']['location']
    end_location = end_geocode[0]['geometry']['location']

    # OSM에서 네트워크 그래프 불러오기 (보행자 경로만 포함)
    G = ox.graph_from_point((start_location['lat'], start_location['lng']), dist=5000, network_type='walk')

    # 각 노드의 고도 데이터를 배치 요청으로 가져오는 함수
    def add_elevation_data_batch(G):
        nodes = list(G.nodes)
        node_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in nodes]

        # 배치 요청으로 고도 정보 가져오기
        elevation_results = []
        for i in range(0, len(node_coords), 512):
            batch = node_coords[i:i + 512]
            elevation_results.extend(gmaps.elevation(batch))

        # 고도 정보를 노드에 추가
        for node, elevation_result in zip(nodes, elevation_results):
            G.nodes[node]['elevation'] = elevation_result['elevation']

    # 그래프에 고도 정보 추가
    add_elevation_data_batch(G)

    # 노드와 엣지에 선호 환경, 경사도, 주변 시설에 따른 가중치를 적용하는 함수
    def apply_weights(G, user_preference):
        for u, v, data in G.edges(data=True):
            # 기본 가중치 설정
            base_weight = data.get('length', 1)
            environment_weight = 1.0
            facility_weight = 1.0
            incline_weight = 1.0

            # 환경에 따른 가중치 조정
            if any(env in data.get('landuse', '') or env in data.get('natural', '') or env in data.get('leisure', '') 
                   for env in user_preference['environment']):
                environment_weight = 0.8  # 선호 환경일 경우 가중치 감소

            # 주변 시설에 따른 가중치 조정
            for facility in user_preference['facilities']:
                if facility in data.get('amenity', ''):
                    facility_weight *= 0.8 # 주변 시설일 경우 가중치 감소

            # 경사도 정보 계산 및 가중치 적용
            elevation_diff = abs(G.nodes[v].get('elevation', 0) - G.nodes[u].get('elevation', 0))
            incline = elevation_diff / data['length'] if data['length'] > 0 else 0  # incline 계산

            if user_preference['incline'] == 'low' and incline > 0.02:  # 조정된 기준 값
                incline_weight = 1.5  # 낮은 경사도 선호 시, 높은 경사도에 큰 패널티 부여
            elif user_preference['incline'] == 'high' and incline < 0.02:
                incline_weight = 1.5  # 높은 경사도 선호 시, 낮은 경사도에 큰 가중치 감소

            # 최종 가중치 계산
            final_weight = base_weight * environment_weight * facility_weight * incline_weight

            # 가중치와 디버깅 정보 출력
            print(f"Edge ({u}, {v}):")
            print(f"  Base Weight = {base_weight}")
            print(f"  Environment Weight = {environment_weight}")
            print(f"  Facility Weight = {facility_weight}")
            print(f"  Incline Weight = {incline_weight}")
            print(f"  Final Weight = {final_weight}")

            # 최종 가중치를 엣지에 할당
            data['weight'] = final_weight

        return G

    # 가중치 적용
    G = apply_weights(G, user_preference)

    # 출발지와 목적지 노드 찾기 (경도, 위도 순서로 입력)
    start_node = ox_distance.nearest_nodes(G, start_location['lng'], start_location['lat'])
    end_node = ox_distance.nearest_nodes(G, end_location['lng'], end_location['lat'])

    # 최적 경로 계산
    optimal_route = nx.shortest_path(G, start_node, end_node, weight='weight')

    # Folium 지도 생성
    m = folium.Map(location=[start_location['lat'], start_location['lng']], zoom_start=14)

    # 경로에 있는 각 노드의 좌표를 가져오기
    route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in optimal_route]

    # 경로를 folium PolyLine으로 지도에 추가
    folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.7).add_to(m)

    # 출발지와 목적지 마커 추가
    folium.Marker([start_location['lat'], start_location['lng']], tooltip="출발지").add_to(m)
    folium.Marker([end_location['lat'], end_location['lng']], tooltip="도착지").add_to(m)

    # HTML 파일로 저장
    m.save("final_optimal_route_map.html")
    print("지도 파일이 optimal_route_map.html로 저장되었습니다.")
    
    m
