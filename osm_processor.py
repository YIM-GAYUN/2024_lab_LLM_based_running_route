import folium
import osmnx as ox
import networkx as nx
from osmnx import distance as ox_distance
from scipy.spatial import cKDTree
import random

# Google Maps API 키 설정
import googlemaps
gmaps = googlemaps.Client(key='your key')

def get_osm_features(start, end, features):
    start_geocode = gmaps.geocode(start, language='ko')
    if not start_geocode:
        raise ValueError("Invalid start location.")
    start_location = start_geocode[0]['geometry']['location']
    
    # 사용자 입력으로부터 도로 유형, 경사도, 환경 가져오기, 값이 제공되지 않았을 경우 기본값 None
    inclination = features.get('Inclination', None)
    environment = features.get('Environment', None)
    distance_goal = features.get('Distance', None)
    additional_preference = features.get('Additional preference', None)

    # OSM에서 네트워크 그래프 불러오기 (보행자 경로만 포함)
    if (distance_goal == None or distance_goal == 'Not specified'): # 목표 거리가 없을 경우 default 3km
        dist = 3000
        G = ox.graph_from_point((start_location['lat'], start_location['lng']), dist=dist, network_type='walk')
    else: # 목표 거리가 있을 경우
        # distance_goal에서 'km'를 제거하고 숫자로 변환 후 1000을 곱함
        dist = int(distance_goal.replace('km', '').strip()) * 1000
        G = ox.graph_from_point((start_location['lat'], start_location['lng']), dist=dist, network_type='walk')

    # 고도 데이터 추가 함수
    def add_elevation_data_batch(G):
        nodes = list(G.nodes)
        node_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in nodes]
        elevation_results = []
        for i in range(0, len(node_coords), 512):
            batch = node_coords[i:i + 512]
            elevation_results.extend(gmaps.elevation(batch))
        for node, elevation_result in zip(nodes, elevation_results):
            G.nodes[node]['elevation'] = elevation_result['elevation']

    # 경사도 가중치 계산 함수
    def apply_incline_weights(G):
        for u, v, data in G.edges(data=True):
            elevation_diff = abs(G.nodes[v].get('elevation', 0) - G.nodes[u].get('elevation', 0))
            incline = elevation_diff / data.get('length', 1)
            if inclination == 'flat' and incline > 0.05:
                data['incline_weight'] = 1.5
            elif inclination == 'inclined' and incline < 0.05:
                data['incline_weight'] = 1.5
            else:
                data['incline_weight'] = 1.0
        return G

    # 시설 가중치 계산 함수
    def compute_nearest_facilities_kdtree(G, facilities_data, threshold=300, min_weight=0.5, max_weight=1.0):
        facility_points = [ # 시설의 중심 좌표 계산
            (geom.centroid.y, geom.centroid.x) if geom.is_valid and hasattr(geom, "centroid") else None
            for geom in facilities_data.geometry
        ]
        facility_points = [point for point in facility_points if point is not None] # 유효한 좌표
        facility_tree = cKDTree(facility_points) if facility_points else None # 시설 좌표에 대한 빠른 거리 계산
        
        # 추가 선호도를 기준으로 가중치 조절
        preferred_tags = {"toilets": 1, "parking": 1, "drinking_water": 1, "convenience": 1}
        if additional_preference:
            # 사용자 선호도에 따라 가중치 설정
            for preference in additional_preference:
                if preference in preferred_tags:
                    preferred_tags[preference] = 0.5 # 선호하는 시설의 가중치를 낮게 설정

        for node, data in G.nodes(data=True):
            node_point = (data['y'], data['x'])
            _, distance = facility_tree.query(node_point) # KD-Tree를 이용해 노드에서 가장 가까운 시설까지의 거리 계산
            if distance <= threshold: # 300m 이내일 경우
                for facility_type, tag in [("toilets", "amenity"), ("parking", "amenity"), ("drinking_water", "amenity"), ("convenience", "shop")]:
                    if facility_type in facilities_data[tag].values:  # 시설 유형 확인
                        data['facility_weight'] = preferred_tags[facility_type]
            else:
                data['facility_weight'] = max_weight
        return G

    # 환경 가중치 계산 함수
    def apply_environment_weights(G, threshold=300):
        environment_mapping = {
            "park": ["leisure:park", "landuse:forest"],
            "water": ["natural:water"]
        }
        preferred_tags = set()
        for env in environment:
            if env in environment_mapping:
                preferred_tags.update(environment_mapping[env])
        environment_data = []
        for tag in preferred_tags:
            key, value = tag.split(":")
            env_nodes = ox.geometries_from_place("Seoul, South Korea", tags={key: value})
            if not env_nodes.empty:
                environment_data.extend([
                    (geom.centroid.y, geom.centroid.x) for geom in env_nodes.geometry
                    if geom.is_valid and hasattr(geom, "centroid")
                ])
        if not environment_data: # 환경 요소 데이터가 없으면 G를 그대로 반환
            return G
        environment_tree = cKDTree(environment_data)

        for node, data in G.nodes(data=True):
            node_point = (data['y'], data['x'])
            _, distance = environment_tree.query(node_point)
            if distance <= threshold:
                data['environment_weight'] = 0.5
            else:
                data['environment_weight'] = 1.0
        return G

    # 최종 가중치 계산 함수
    def calculate_final_weights(G):
        for u, v, data in G.edges(data=True):
            data['weight'] = data.get('incline_weight', 1.0) + data.get('facility_weight', 1.0) + data.get('environment_weight', 1.0)
        return G
    
    # 회귀 경로 탐색 함수
    def find_round_trip_routes(G, start_node, target_distance=1500, tolerance=500):
        """
        출발지 → 도착지 → 출발지 회귀 경로를 찾는 함수.
        동일한 경로를 피하면서 왕복 경로를 구성.
        """
        lengths, paths = nx.single_source_dijkstra(G, source=start_node, weight='length')
        # 왕복 경로를 저장할 리스트
        round_trip_routes = []
        
        for target_node, forward_distance in lengths.items():
            if not (target_distance / 2 - tolerance <= forward_distance <= target_distance / 2 + tolerance):
                continue

            # Forward path: 출발지 → 도착지 경로
            forward_path = paths[target_node]

            # 그래프 복사 후 forward path 간선 제거
            G_temp = G.copy()
            for u, v in zip(forward_path[:-1], forward_path[1:]):
                if G_temp.has_edge(u, v):
                    G_temp.remove_edge(u, v)
                if G_temp.has_edge(v, u):
                    G_temp.remove_edge(v, u)

            try:
                # Backward path: 도착지 → 출발지 경로
                backward_path = nx.shortest_path(G_temp, source=target_node, target=start_node, weight='length')
                backward_distance = sum(G[u][v][0]['length'] for u, v in zip(backward_path[:-1], backward_path[1:]))
                
                # 왕복 거리 계산
                total_distance = forward_distance + backward_distance
                if target_distance - tolerance <= total_distance <= target_distance + tolerance:
                    # 왕복 거리가 조건을 만족하는 경우에만 추가
                    round_trip_routes.append((forward_path, forward_distance, backward_path, backward_distance))
            
            except nx.NetworkXNoPath:
                # 도착지에서 출발지로 가는 경로가 없을 경우 무시
                continue

        if not round_trip_routes:
            raise ValueError("해당하는 왕복 경로를 찾을 수 없습니다.")
        
        return round_trip_routes
    
    def calculate_round_trip_weight(G, forward_path, backward_path):
        forward_weights = [
            G.nodes[node].get('incline_weight', 1.0) +
            G.nodes[node].get('facility_weight', 1.0) +
            G.nodes[node].get('environment_weight', 1.0)
            for node in forward_path
        ]
        backward_weights = [
            G.nodes[node].get('incline_weight', 1.0) +
            G.nodes[node].get('facility_weight', 1.0) +
            G.nodes[node].get('environment_weight', 1.0)
            for node in backward_path
        ]
        print("sum(forward_weights):", sum(forward_weights))
        print("len(backward_weights):",len(backward_weights))
        print("sum(forward_weights) + sum(backward_weights):", sum(forward_weights) + sum(backward_weights))
        print("len(forward_weights) + len(backward_weights):", len(forward_weights) + len(backward_weights))

        return (sum(forward_weights) + sum(backward_weights)) / (len(forward_weights) + len(backward_weights))

    # 출발 노드 설정
    start_node = ox_distance.nearest_nodes(G, start_location['lng'], start_location['lat'])

    # 고도 데이터 추가
    add_elevation_data_batch(G)

    # 시설 데이터 가져오기
    facilities_data = ox.geometries_from_place("Seoul, South Korea", tags={"amenity": ["toilets", "parking", "drinking_water"], "shop": "convenience"})

    # 가중치 계산 적용
    G = apply_incline_weights(G)
    G = compute_nearest_facilities_kdtree(G, facilities_data, threshold=300)
    G = apply_environment_weights(G, threshold=300)
    G = calculate_final_weights(G)

    # 경로 탐색
    try:
        round_trip_routes = find_round_trip_routes(G, start_node, target_distance=dist)
    except ValueError as e:
        print(e)
        exit()

    # 왕복 경로별 최종 가중치 계산
    round_trip_weights = []
    for idx, (forward_path, forward_distance, backward_path, backward_distance) in enumerate(round_trip_routes):
        weight = calculate_round_trip_weight(G, forward_path, backward_path)
        print("최종 weight:",weight)
        total_distance = forward_distance + backward_distance
        round_trip_weights.append((idx, weight, total_distance, forward_path, backward_path))
        
    # 최종 가중치가 낮은 상위 3개의 경로 선택
    top_3_round_trips = sorted(round_trip_weights, key=lambda x: x[1])[:3]

    # 색상 목록 (각 경로별 색상 설정)
    colors = ['red', 'green', 'blue']

    # Folium 지도 생성
    m = folium.Map(location=[start_location['lat'], start_location['lng']], zoom_start=14)
    folium.Marker([start_location['lat'], start_location['lng']], tooltip="출발지", icon=folium.Icon(color="black")).add_to(m)

    # 상위 3개 왕복 경로를 지도에 추가
    for idx, (route_idx, weights, total_distance, forward_path, backward_path) in enumerate(top_3_round_trips):
        forward_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in forward_path]
        folium.PolyLine(
            forward_coords,
            color=colors[idx],
            weight=5,
            opacity=0.7,
            tooltip=f"Forward Route {idx + 1} (Weight: {weights:.2f}, Distance: {total_distance:.2f}m)"
        ).add_to(m)

        backward_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in backward_path]
        folium.PolyLine(
            backward_coords,
            color=colors[idx],
            weight=5,
            opacity=0.7,
            dash_array="5, 5",
            tooltip=f"Backward Route {idx + 1} (Weight: {weights:.2f}, Distance: {total_distance:.2f}m)"
        ).add_to(m)

        target_node = forward_path[-1]
        target_lat, target_lon = G.nodes[target_node]['y'], G.nodes[target_node]['x']
        folium.Marker(
            [target_lat, target_lon],
            tooltip=f"Destination {idx + 1}",
            icon=folium.Icon(color=colors[idx])
        ).add_to(m)

    # 지도 저장
    m.save("top_3_round_trip_routes.html")
    print("지도 파일이 top_3_round_trip_routes.html로 저장되었습니다.")

    # 최종 결과 출력
    print("\nTop 3 Round Trip Routes with Lowest Weights:")
    for idx, (route_idx, weights, total_distance, forward_path, backward_path) in enumerate(top_3_round_trips):
        print(f"\nRoute {idx + 1}:")
        print(f"  Original Index: {route_idx}")
        print(f"  Final Weight: {weights:.2f}")
        print(f"  Total Distance: {total_distance:.2f}m")
        print(f"  Forward Path: {forward_path}")
        print(f"  Backward Path: {backward_path}")
        
    print(f'dist: {dist}')
    
    
