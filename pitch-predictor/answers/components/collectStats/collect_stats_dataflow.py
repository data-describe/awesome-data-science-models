# libraries
from __future__ import print_function
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
import apache_beam as beam
import argparse
import datetime
import logging
import os
import uuid


# parallel functions
from ParDoFns.gameday import GameDay
from ParDoFns.collectGames import collectGames
from ParDoFns.collectPitchers import collectPitchers
from ParDoFns.collectStats import collectStats

# environment vars
# os.environ['GOOGLE_APPLICATION_CREDENTIALS']='{{ GCP_PROJECT }}-da165d24798a.json'


# create a list of days to collect
today = datetime.datetime.now().date()
start_date = datetime.datetime.strptime(
    "20190801", "%Y%m%d"
).date()  # opening day 2019 :: 20190328

pcol = []
while start_date < today:
    day = str(start_date.day)
    month = str(start_date.month)
    year = str(start_date.year)

    gday = GameDay(day, month, year)
    pcol.append(gday)

    start_date += datetime.timedelta(days=1)


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runner",
        dest="runner",
        default="DataflowRunner",
        help="Select between DirectRunner vs DataflowRunner",
    )
    parser.add_argument(
        "--project",
        dest="project",
        default="mwpmltr",
        help="Select the gcp project to run this job",
    )
    parser.add_argument(
        "--region",
        dest="region",
        default="us-central1",
        help="Select the gcp project to run this job",
    )
    parser.add_argument(
        "--staging_location",
        dest="staging_location",
        default=f"gs://{{ GCP_PROJECT }}-dataflow/dataflow_stage/",
        help="Select the staging location for this job",
    )
    parser.add_argument(
        "--temp_location",
        dest="temp_location",
        default=f"gs://{{ GCP_PROJECT }}-dataflow/dataflow_tmp/",
        help="Select the temp location for this job",
    )
    parser.add_argument(
        "--setup_file",
        dest="setup_file",
        default="./setup.py",
        help="Config options for the pipeline",
    )
    parser.add_argument(
        "--year",
        dest="year",
        action="append",
        default=["2019"],
        help="Calendar years to pull data",
    )

    known_args, pipeline_args = parser.parse_known_args(argv)
    pipeline_args.extend(
        [
            "--runner=" + known_args.runner,
            "--project=" + known_args.project,
            "--region=" + known_args.region,
            "--staging_location=" + known_args.staging_location,
            "--temp_location=" + known_args.temp_location,
            "--job_name=mlb-collect-games-{}".format((str(uuid.uuid4()))[0:6]),
            "--setup_file=" + known_args.setup_file,
        ]
    )

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    logging.info(pipeline_args)

    # define schema for BigQuery sink
    schema = "dateStamp:DATE,park_sv_id:STRING,play_guid:STRING,ab_total:FLOAT64,ab_count:FLOAT64,pitcher_id:STRING,batter_id:STRING,ab_id:FLOAT64,des:STRING,type:STRING,id:STRING,sz_top:FLOAT64,sz_bot:FLOAT64,pfx_xDataFile:FLOAT64,pfx_zDataFile:FLOAT64,mlbam_pitch_name:STRING,zone_location:FLOAT64,pitch_con:FLOAT64,stand:STRING,strikes:FLOAT64,balls:FLOAT64,p_throws:STRING,gid:STRING,pdes:STRING,spin:FLOAT64,norm_ht:FLOAT64,inning:FLOAT64,pitcher_team:STRING,tstart:FLOAT64,vystart:FLOAT64,ftime:FLOAT64,pfx_x:FLOAT64,pfx_z:FLOAT64,uncorrected_pfx_x:FLOAT64,uncorrected_pfx_z:FLOAT64,x0:FLOAT64,y0:FLOAT64,z0:FLOAT64,vx0:FLOAT64,vy0:FLOAT64,vz0:FLOAT64,ax:FLOAT64,ay:FLOAT64,az:FLOAT64,start_speed:FLOAT64,px:FLOAT64,pz:FLOAT64,pxold:FLOAT64,pzold:FLOAT64,tm_spin:FLOAT64,sb:FLOAT64"

    # Init day for each year
    pcol = []
    for year in known_args.year:
        day = "1"
        month = "1"
        year = str(year)

        gday = GameDay(day, month, year)
        pcol.append(gday)

    # begin pipeline
    with beam.Pipeline(options=pipeline_options) as p:
        years = p | "Create Date Objects" >> beam.Create(pcol)
        pitchers = years | "Collect Pitchers" >> beam.ParDo(collectPitchers())
        games = pitchers | "Collect Games" >> beam.ParDo(collectGames())
        stats = games | "Collect Stats" >> beam.ParDo(collectStats())
        stats | "Write to BigQuery" >> beam.io.WriteToBigQuery(
            table="raw_games",
            dataset="baseball",
            project="{{ GCP_PROJECT }}",
            create_disposition="CREATE_IF_NEEDED",
            write_disposition="WRITE_APPEND",
            schema=schema,
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()

