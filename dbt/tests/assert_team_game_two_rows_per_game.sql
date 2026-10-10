-- Assert that each game in fct_team_game_epa has exactly 2 rows and exactly 1 home team
-- Returns rows that violate the constraint (should be empty)

select
    game_id
from {{ ref('fct_team_game_epa') }}
group by game_id
having count(*) != 2 or sum(is_home) != 1
