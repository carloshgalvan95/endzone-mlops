{{
    config(
        materialized='table',
        tags=['marts', 'team_game', 'epa']
    )
}}

/*
Fact table: team-game level EPA and play metrics.

Grain: One row per team per game (two rows per game, COMPLETED GAMES ONLY).
Purpose: Track offensive and defensive performance metrics for each team in each game.

Filter: Only games with both home_score and away_score not null (completed games).

Key metrics:
- EPA (Expected Points Added) - offensive and defensive totals and averages
- Play counts by type (pass, rush, total)
- Success rates
- Clean EPA and clean success rate (pass and run only, no kneel, spike, or garbage time)
- Scoring and game context

The original EPA columns still average every play = 1 row. Clean columns are
additional. They do not replace those averages.

Franchise mapping: Handles team relocations (OAK->LV, SD->LAC, STL->LA) by joining
pbp to games on franchise_abbr. team/opponent columns preserve historical names from
games.parquet; team_franchise/opponent_franchise show current franchise abbreviations.

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
    where home_score is not null
      and away_score is not null

),

franchise_map as (

    select
        team_abbr,
        franchise_abbr
    from {{ ref('team_franchise_map') }}

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
        rush_attempt,
        qb_kneel,
        qb_spike,
        win_probability
    from {{ ref('stg_nflverse__pbp') }}
    where play = 1

),

/*
Clean plays are a subset of play = 1.
play_type in ('pass', 'run') drops no_play and every other play type.
qb_kneel and qb_spike drop flagged kneels and spikes that still have
play_type pass or run. A null flag is not a kneel or a spike.
Garbage time is possession-team wp (win_probability) below 0.05 or above
0.95 at the snap. Those two cutoffs are fixed. A null wp is not garbage.
*/
clean_pbp as (

    select
        game_id,
        possession_team,
        defensive_team,
        epa,
        success
    from pbp
    where play_type in ('pass', 'run')
      and coalesce(qb_kneel, 0) = 0
      and coalesce(qb_spike, 0) = 0
      and (
          win_probability is null
          or (
              win_probability >= 0.05
              and win_probability <= 0.95
          )
      )

),

offensive_stats as (

    select
        game_id,
        possession_team as franchise_abbr,
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
        defensive_team as franchise_abbr,
        sum(epa) as defensive_epa_allowed,
        avg(epa) as avg_defensive_epa_allowed
    from pbp
    group by game_id, defensive_team

),

clean_offensive_stats as (

    select
        game_id,
        possession_team as franchise_abbr,
        count(*) as clean_offensive_plays,
        avg(epa) as avg_clean_offensive_epa,
        avg(case when success = 1 then 1.0 else 0.0 end) as clean_success_rate
    from clean_pbp
    group by game_id, possession_team

),

clean_defensive_stats as (

    select
        game_id,
        defensive_team as franchise_abbr,
        count(*) as clean_defensive_plays,
        avg(epa) as avg_clean_defensive_epa_allowed,
        avg(case when success = 1 then 1.0 else 0.0 end) as clean_defensive_success_rate
    from clean_pbp
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
        -1 * g.spread_line as spread_line,
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
        
        -- Franchise columns for grouping across relocations
        team_fm.franchise_abbr as team_franchise,
        opp_fm.franchise_abbr as opponent_franchise,
        
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
        d.avg_defensive_epa_allowed,

        -- Clean EPA. Null average means the team had no clean plays.
        coalesce(co.clean_offensive_plays, 0) as clean_offensive_plays,
        co.avg_clean_offensive_epa,
        co.clean_success_rate,
        coalesce(cd.clean_defensive_plays, 0) as clean_defensive_plays,
        cd.avg_clean_defensive_epa_allowed,
        cd.clean_defensive_success_rate

    from team_games tg
    left join franchise_map team_fm
        on tg.team = team_fm.team_abbr
    left join franchise_map opp_fm
        on tg.opponent = opp_fm.team_abbr
    left join offensive_stats o
        on tg.game_id = o.game_id
        and team_fm.franchise_abbr = o.franchise_abbr
    left join defensive_stats d
        on tg.game_id = d.game_id
        and team_fm.franchise_abbr = d.franchise_abbr
    left join clean_offensive_stats co
        on tg.game_id = co.game_id
        and team_fm.franchise_abbr = co.franchise_abbr
    left join clean_defensive_stats cd
        on tg.game_id = cd.game_id
        and team_fm.franchise_abbr = cd.franchise_abbr

)

select * from final
