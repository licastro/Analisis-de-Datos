# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:35:17 2024

@author: Lucía
"""

#%%

import duckdb
import xml.etree.ElementTree as ET
import pandas as pd

#%%

# Función para convertir XML a diccionario
def xml_to_dict(element):
    result = {}
    if element.tag == "goal":
        result["values"] = []
        for value in element.findall('value'):
            value_dict = xml_to_dict(value)
            result["values"].append(value_dict)
    else:
        for child in element:
            result[child.tag] = xml_to_dict(child) if len(child) > 0 else child.text
    return result

# Función para aplicar a cada celda de la columna 'goal'
def parse_goal(goal_string):
    if pd.isna(goal_string):
        return None
    root = ET.fromstring(goal_string)
    return xml_to_dict(root)

# Función para extraer las claves deseadas y crear nuevas filas
def extraer_claves(goal_dict, x, y):
    if pd.isna(goal_dict) or 'values' not in goal_dict:
        return [{'id': None, 'elapsed': None, 'player_1': None, 'team': None}]

    # Lista para almacenar los registros
    records = []

    # Recorre cada evento en 'values'
    for event in goal_dict['values']:
        records.append({
            'id': event.get('id'),
            'elapsed': event.get('elapsed'),
            'player1': event.get('player1', None),
            'team': event.get('team'),
            'tipo': event.get('goal_type'),
            'date': x,
            'stage': y
        })

    return records

#%%

url = "https://auto-scan.com.ar/labdatos/"

equipos_crudo = pd.read_csv(url+"enunciado_equipos.csv")
jugadores_crudo = pd.read_csv(url+"enunciado_jugadores.csv")
jugadores_atributos_crudo = pd.read_csv(url+"enunciado_jugadores_atributos.csv")
ligas_crudo = pd.read_csv(url+"enunciado_liga.csv")
paises_crudo = pd.read_csv(url+"enunciado_paises.csv")
partidos_crudo = pd.read_csv(url+"enunciado_partidos.csv")

#%%

partidos_crudo2= partidos_crudo.copy()

partidos_crudo2.loc[:, 'goal_dict'] = partidos_crudo2['goal'].apply(parse_goal)

resultados = partidos_crudo2.apply(lambda row: extraer_claves(row['goal_dict'], row['date'], row['stage']), axis=1)

goles_crudo = pd.DataFrame([item for sublist in resultados for item in sublist])

#%%

equipo = duckdb.sql('''SELECT team_api_id AS id_equipo, team_long_name AS nombre_equipo
                       FROM equipos_crudo;''').df()

#%%

jugador = duckdb.sql('''SELECT player_api_id AS id, player_name AS nombre, birthday AS fecha_nacimiento
                        FROM jugadores_crudo;''').df()

#%%

pais = duckdb.sql('''SELECT id AS id_pais, name AS nombre_pais
                     FROM paises_crudo;''').df()

#%%

liga = duckdb.sql('''SELECT country_id AS id_liga ,name AS nombre_liga,country_id AS id_pais
                     FROM ligas_crudo;''').df()

#%%

partido = duckdb.sql('''SELECT match_api_id AS id_partido, date AS fecha_partido, home_team_api_id AS equipo_local,
                               away_team_api_id AS equipo_visitante, home_team_goal AS goles_local, away_team_goal AS goles_visitante
                        FROM partidos_crudo ''').df()

#%%

temporada = duckdb.sql('''SELECT DISTINCT CONCAT(country_id, '-', season) AS id_season,
                          MIN(date) AS fecha_inicio, MAX(date) AS fecha_fin
                          FROM partidos_crudo
                          GROUP BY season, country_id;''').df()

#%%

partidos_jugadores_auxiliar = duckdb.sql('''SELECT country_id, season, home_team_api_id AS team_api_id, date, home_player_1 AS player_id,
                                            CONCAT(country_id, '-', season, '-', home_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, home_team_api_id, date, home_player_2 AS player_id,
                                            CONCAT(country_id, '-', season, '-', home_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, home_team_api_id, date, home_player_3 AS player_id,
                                            CONCAT(country_id, '-', season, '-', home_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, home_team_api_id, date, home_player_4 AS player_id,
                                            CONCAT(country_id, '-', season, '-', home_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, home_team_api_id, date, home_player_5 AS player_id,
                                            CONCAT(country_id, '-', season, '-', home_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, home_team_api_id, date, home_player_6 AS player_id,
                                            CONCAT(country_id, '-', season, '-', home_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, home_team_api_id, date, home_player_7 AS player_id,
                                            CONCAT(country_id, '-', season, '-', home_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, home_team_api_id, date, home_player_8 AS player_id,
                                            CONCAT(country_id, '-', season, '-', home_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, home_team_api_id, date, home_player_9 AS player_id,
                                            CONCAT(country_id, '-', season, '-', home_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, home_team_api_id, date, home_player_10 AS player_id,
                                            CONCAT(country_id, '-', season, '-', home_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, home_team_api_id, date, home_player_11 AS player_id,
                                            CONCAT(country_id, '-', season, '-', home_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, away_team_api_id AS team_api_id, date, away_player_1 AS player_id,
                                            CONCAT(country_id, '-', season, '-', away_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, away_team_api_id, date, away_player_2 AS player_id,
                                            CONCAT(country_id, '-', season, '-', away_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, away_team_api_id, date, away_player_3 AS player_id,
                                            CONCAT(country_id, '-', season, '-', away_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, away_team_api_id, date, away_player_4 AS player_id,
                                            CONCAT(country_id, '-', season, '-', away_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, away_team_api_id, date, away_player_5 AS player_id,
                                            CONCAT(country_id, '-', season, '-', away_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, away_team_api_id, date, away_player_6 AS player_id,
                                            CONCAT(country_id, '-', season, '-', away_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, away_team_api_id, date, away_player_7 AS player_id,
                                            CONCAT(country_id, '-', season, '-', away_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, away_team_api_id, date, away_player_8 AS player_id,
                                            CONCAT(country_id, '-', season, '-', away_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, away_team_api_id, date, away_player_9 AS player_id,
                                            CONCAT(country_id, '-', season, '-', away_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, away_team_api_id, date, away_player_10 AS player_id,
                                            CONCAT(country_id, '-', season, '-', away_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo
                                            UNION
                                            SELECT country_id, season, away_team_api_id, date, away_player_11 AS player_id,
                                            CONCAT(country_id, '-', season, '-', away_team_api_id) AS id_plantel,
                                            CONCAT(country_id, '-', season) AS id_season
                                            FROM partidos_crudo''').df()

#%%

plantel = duckdb.sql('''SELECT id_plantel, COUNT(DISTINCT player_id) AS cantidad_jugadores
                        FROM partidos_jugadores_auxiliar AS p
                        WHERE p.date >= (SELECT fecha_inicio FROM temporada WHERE id_season = p.id_season)
                        AND p.date <= (SELECT fecha_fin FROM temporada WHERE id_season = p.id_season)
                        GROUP BY id_plantel''').df()

#%%

caracteristicas = duckdb.sql('''SELECT DISTINCT id_plantel,
                                jac.player_api_id AS id_jugador, jac.sprint_speed AS velocidad, jac.shot_power AS potencia,
                                jac.preferred_foot AS pie_dominante, jac.ball_control AS tecnica, jac.overall_rating,
                                jc.height AS altura, jc.weight AS peso
                                FROM partidos_jugadores_auxiliar AS pja
                                INNER JOIN temporada ON pja.id_season = temporada.id_season
                                INNER JOIN jugadores_atributos_crudo AS jac ON pja.player_id = jac.player_api_id
                                INNER JOIN jugadores_crudo AS jc ON jac.player_api_id = jc.player_api_id
                                WHERE pja.date >= temporada.fecha_inicio AND pja.date <= temporada.fecha_fin
                                AND jac.date >= temporada.fecha_inicio AND jac.date <= temporada.fecha_fin
                                ORDER BY id_plantel, pja.date;''').df()

#%%

goles = duckdb.sql('''SELECT id AS id_gol, date AS fecha_gol, stage AS ronda, elapsed AS tiempo,
                   player1 AS jugador_que_anoto, team AS equipo_que_anoto, tipo AS tipo_gol
                      FROM goles_crudo;''').df()

#%%

pertenece_a_liga = duckdb.sql('''SELECT DISTINCT home_team_api_id AS id_equipo, league_id AS id_liga
                                 FROM partidos_crudo''').df()

#%%

local = duckdb.sql('''SELECT match_api_id AS id_partido, home_team_api_id AS id_equipo
                      FROM partidos_crudo''').df()

#%%

visitante = duckdb.sql('''SELECT match_api_id AS id_partido, away_team_api_id AS id_equipo
                          FROM partidos_crudo''').df()

#%%

equipo_anota_gol = duckdb.sql('''SELECT id_gol, equipo_que_anoto AS id_equipo
                                 FROM goles''').df()

#%%

equipo_forma_plantel = duckdb.sql('''SELECT id_plantel, split_part(id_plantel, '-', 3) AS id_equipo
                                     FROM plantel''').df()

#%%

pertenece_a_pais = duckdb.sql('''SELECT DISTINCT league_id AS id_liga, country_id AS id_pais
                                 FROM partidos_crudo''').df()


#%%

se_juega_en_liga = duckdb.sql('''SELECT match_api_id AS id_partido, league_id as id_liga
                                 FROM partidos_crudo''').df()

#%%

partido_tiene_goles = duckdb.sql('''SELECT id_gol, id_partido
                                    FROM goles
                                    INNER JOIN partido ON fecha_gol = fecha_partido
                                    WHERE equipo_visitante = equipo_que_anoto OR equipo_local = equipo_que_anoto''').df()

#%%

partido_en_temporada = duckdb.sql('''SELECT match_api_id AS id_partido, CONCAT(country_id, '-', season) AS id_season
                                     FROM partidos_crudo''').df()

#%%

plantel_de_la_temporada = duckdb.sql('''SELECT DISTINCT id_plantel, id_season
                                        FROM partidos_jugadores_auxiliar''').df()

#%%

jugador_en_plantel = duckdb.sql('''SELECT DISTINCT id_jugador, id_plantel
                                   FROM caracteristicas''').df()

#%%

jugador_hace_gol = duckdb.sql('''SELECT id_gol, jugador_que_anoto AS id_jugador
                                 FROM goles''').df()
                                 
#%%

pais.to_csv('pais.csv', index=False)
liga.to_csv('liga.csv', index=False)
equipo.to_csv('equipo.csv', index=False)
partido.to_csv('partido.csv', index=False)
goles.to_csv('goles.csv', index=False)
temporada.to_csv('temporada.csv', index=False)
plantel.to_csv('plantel.csv', index=False)
caracteristicas.to_csv('caracteristicas.csv', index=False)
jugador.to_csv('jugador.csv', index=False)
equipo_anota_gol.to_csv('equipo_anota_gol.csv', index=False)
equipo_forma_plantel.to_csv('equipo_forma_plantel.csv', index=False)
jugador_en_plantel.to_csv('jugador_en_plantel.csv', index=False)
jugador_hace_gol.to_csv('jugador_hace_gol.csv', index=False)
local.to_csv('local.csv', index=False)
partido_en_temporada.to_csv('partido_en_temporada.csv', index=False)
partido_tiene_goles.to_csv('partido_tiene_goles.csv', index=False)
pertenece_a_liga.to_csv('pertenece_a_liga.csv', index=False)
pertenece_a_pais.to_csv('pertenece_a_pais.csv', index=False)
plantel_de_la_temporada.to_csv('plantel_de_la_temporada.csv', index=False)
se_juega_en_liga.to_csv('se_juega_en_liga.csv', index=False)
visitante.to_csv('visitante.csv', index=False)
