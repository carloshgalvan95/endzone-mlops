{{
    config(
        materialized='table',
        tags=['marts', 'team_game', 'epa']
    )
}}

/*
Fact table: team-game level EPA and play metrics.

Grain: One row per team per game (two rows per game).
Purpose: Track offensive and defensive performance metrics for each team in each game.

Key metrics:
- EPA (Expected Points Added) - offensive and defensive totals and averages
- Play counts by type (pass, rush, total)
- Success rates
- Scoring and game context

Useful for:
- Pregame prediction features (recent team performance)
- Model training (offensive/defensive strength indicators)
- Performance analysis and feature engineering
*/

with games as (

    select
        game_id,
        season,
        week,
        game_type,
        away_team,
        home_team,
        away_score,
        home_score,
        spread_line,
        total_line,
        gameday
    from {{ ref('stg_nflverse__games') }}

),

pbp as (

    select
        game_id,
        possession_team,
        defensive_team,
        play,
        play_type,
        epa,
        success,
        pass_attempt,
        rush_attempt
    from {{ ref('stg_nflverse__pbp') }}
    where play = 1  -- Only count valid plays (exclude no-plays, penalties without play)

),

offensive_stats as (

    select
        game_id,
        possession_team as team,
        count(*) as offensive_plays,
        sum(case when pass_attempt = 1 then 1 else 0 end) as pass_plays,
        sum(case when rush_attempt = 1 then 1 else 0 end) as rush_plays,
        sum(epa) as offensive_epa,
        avg(epa) as avg_offensive_epa,
        sum(case when pass_attempt = 1 then epa else 0 end) as pass_epa,
        sum(case when rush_attempt = 1 then epa else 0 end) as rush_epa,
        sum(case when success = 1 then 1 else 0 end) as successful_plays,
        avg(case when success = 1 then 1.0 else 0.0 end) as success_rate
    from pbp
    group by game_id, possession_team

),

defensive_stats as (

    select
        game_id,
        defensive_team as team,
        sum(epa) as defensive_epa_allowed,
        avg(epa) as avg_defensive_epa_allowed
    from pbp
    group by game_id, defensive_team

),

team_games as (

    -- Home team records
    select
        g.game_id,
        g.season,
        g.week,
        g.game_type,
        g.home_team as team,
        g.away_team as opponent,
        1 as is_home,
        g.home_score as team_score,
        g.away_score as opponent_score,
        g.spread_line,
        g.total_line,
        g.gameday
    from games g

    union all

    -- Away team records
    select
        g.game_id,
        g.season,
        g.week,
        g.game_type,
        g.away_team as team,
        g.home_team as opponent,
        0 as is_home,
        g.away_score as team_score,
        g.home_score as opponent_score,
        -1 * g.spread_line as spread_line,  -- Flip spread for away team
        g.total_line,
        g.gameday
    from games g

),

final as (

    select
        tg.game_id,
        tg.season,
        tg.week,
        tg.game_type,
        tg.team,
        tg.opponent,
        tg.is_home,
        tg.team_score,
        tg.opponent_score,
        tg.spread_line,
        tg.total_line,
        tg.gameday,
        
        -- Offensive metrics
        coalesce(o.offensive_plays, 0) as offensive_plays,
        coalesce(o.pass_plays, 0) as pass_plays,
        coalesce(o.rush_plays, 0) as rush_plays,
        coalesce(o.offensive_epa, 0.0) as offensive_epa,
        o.avg_offensive_epa,
        coalesce(o.pass_epa, 0.0) as pass_epa,
        coalesce(o.rush_epa, 0.0) as rush_epa,
        coalesce(o.successful_plays, 0) as successful_plays,
        o.success_rate,
        
        -- Defensive metrics (EPA allowed)
        coalesce(d.defensive_epa_allowed, 0.0) as defensive_epa_allowed,
        d.avg_defensive_epa_allowed

    from team_games tg
    left join offensive_stats o
        on tg.game_id = o.game_id
        and tg.team = o.team
    left join defensive_stats d
        on tg.game_id = d.game_id
        and tg.team = d.team

)

select * from final
