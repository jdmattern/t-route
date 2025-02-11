import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import troute.nhd_io as nhd_io
from build_tests import parity_check


def nwm_output_generator(
    run,
    results,
    supernetwork_parameters,
    output_parameters,
    parity_parameters,
    restart_parameters,
    parity_set,
    qts_subdivisions,
    return_courant,
    showtiming=False,
    verbose=False,
    debuglevel=0,
    data_assimilation_parameters=False,
    lastobs_df=None,
    link_gage_df=None,
):

    dt = run.get("dt")
    nts = run.get("nts")
    t0 = run.get("t0")

    if parity_parameters:
        parity_check_waterbody_file = parity_parameters.get("parity_check_waterbody_file", None)
        parity_check_file = parity_parameters.get("parity_check_file", None)
        parity_check_input_folder = parity_parameters.get("parity_check_input_folder", None)
        parity_check_file_index_col = parity_parameters.get(
            "parity_check_file_index_col", None
        )
        parity_check_file_value_col = parity_parameters.get(
            "parity_check_file_value_col", None
        )
        parity_check_compare_node = parity_parameters.get("parity_check_compare_node", None)

        # TODO: find a better way to deal with these defaults and overrides.
        parity_set["parity_check_waterbody_file"] = parity_set.get(
            "parity_check_waterbody_file", parity_check_waterbody_file
        )
        parity_set["parity_check_file"] = parity_set.get(
            "parity_check_file", parity_check_file
        )
        parity_set["parity_check_input_folder"] = parity_set.get(
            "parity_check_input_folder", parity_check_input_folder
        )
        parity_set["parity_check_file_index_col"] = parity_set.get(
            "parity_check_file_index_col", parity_check_file_index_col
        )
        parity_set["parity_check_file_value_col"] = parity_set.get(
            "parity_check_file_value_col", parity_check_file_value_col
        )
        parity_set["parity_check_compare_node"] = parity_set.get(
            "parity_check_compare_node", parity_check_compare_node
        )

    ################### Output Handling
    if showtiming:
        start_time = time.time()
    if verbose:
        print(f"Handling output ...")

    csv_output = output_parameters.get("csv_output", None)
    csv_output_folder = None
    rsrto = output_parameters.get("hydro_rst_output", None)
    chrto = output_parameters.get("chrtout_output", None)

    if csv_output:
        csv_output_folder = output_parameters["csv_output"].get(
            "csv_output_folder", None
        )
        csv_output_segments = csv_output.get("csv_output_segments", None)

    if (debuglevel <= -1) or csv_output_folder or rsrto or chrto:

        qvd_columns = pd.MultiIndex.from_product(
            [range(nts), ["q", "v", "d"]]
        ).to_flat_index()

        flowveldepth = pd.concat(
            [pd.DataFrame(r[1], index=r[0], columns=qvd_columns) for r in results],
            copy=False,
        )

        if return_courant:
            courant_columns = pd.MultiIndex.from_product(
                [range(nts), ["cn", "ck", "X"]]
            ).to_flat_index()
            courant = pd.concat(
                [
                    pd.DataFrame(r[2], index=r[0], columns=courant_columns)
                    for r in results
                ],
                copy=False,
            )

    if rsrto:

        if verbose:
            print("- writing restart files")

        wrf_hydro_restart_dir = rsrto.get(
            "wrf_hydro_channel_restart_source_directory", None
        )
        wrf_hydro_restart_write_dir = rsrto.get(
            "wrf_hydro_channel_restart_output_directory", wrf_hydro_restart_dir
        )
        if wrf_hydro_restart_dir:

            wrf_hydro_channel_restart_new_extension = rsrto.get(
                "wrf_hydro_channel_restart_new_extension", "TRTE"
            )

            if rsrto.get("wrf_hydro_channel_restart_pattern_filter", None):
                # list of WRF Hydro restart files
                wrf_hydro_restart_files = sorted(
                    Path(wrf_hydro_restart_dir).glob(
                        rsrto["wrf_hydro_channel_restart_pattern_filter"]
                        + "[!"
                        + wrf_hydro_channel_restart_new_extension
                        + "]"
                    )
                )
            else:
                wrf_hydro_restart_files = sorted(
                    Path(wrf_hydro_restart_dir) / f for f in run["qlat_files"]
                )

            if len(wrf_hydro_restart_files) > 0:
                nhd_io.write_channel_restart_to_wrf_hydro(
                    flowveldepth,
                    wrf_hydro_restart_files,
                    Path(wrf_hydro_restart_write_dir),
                    restart_parameters.get("wrf_hydro_channel_restart_file"),
                    dt,
                    nts,
                    t0,
                    restart_parameters.get("wrf_hydro_channel_ID_crosswalk_file"),
                    restart_parameters.get(
                        "wrf_hydro_channel_ID_crosswalk_file_field_name"
                    ),
                    wrf_hydro_channel_restart_new_extension,
                )
            else:
                # print error and/or raise exception
                str = "WRF Hydro restart files not found - Aborting restart write sequence"
                raise AssertionError(str)

    if chrto:
        
        if verbose:
            print("- writing results to CHRTOUT")
        
        chrtout_read_folder = chrto.get(
            "wrf_hydro_channel_output_source_folder", None
        )
        chrtout_write_folder = chrto.get(
            "wrf_hydro_channel_final_output_folder", chrtout_read_folder
        )
        if chrtout_read_folder:
            wrf_hydro_channel_output_new_extension = chrto.get(
                "wrf_hydro_channel_output_new_extension", "TRTE"
            )

            chrtout_files = sorted(
                Path(chrtout_read_folder) / f for f in run["qlat_files"]
            )

            nhd_io.write_q_to_wrf_hydro(
                flowveldepth,
                chrtout_files,
                Path(chrtout_write_folder),
                qts_subdivisions,
                wrf_hydro_channel_output_new_extension,
            )

    if csv_output_folder: 
    
        if verbose:
            print("- writing flow, velocity, and depth results to .csv")

        # create filenames
        # TO DO: create more descriptive filenames
        if supernetwork_parameters.get("title_string", None):
            filename_fvd = (
                "flowveldepth_" + supernetwork_parameters["title_string"] + ".csv"
            )
            filename_courant = (
                "courant_" + supernetwork_parameters["title_string"] + ".csv"
            )
        else:
            run_time_stamp = datetime.now().isoformat()
            filename_fvd = "flowveldepth_" + run_time_stamp + ".csv"
            filename_courant = "courant_" + run_time_stamp + ".csv"

        output_path = Path(csv_output_folder).resolve()

        # no csv_output_segments are specified, then write results for all segments
        if not csv_output_segments:
            csv_output_segments = flowveldepth.index
        
        flowveldepth = flowveldepth.sort_index()
        flowveldepth.loc[csv_output_segments].to_csv(output_path.joinpath(filename_fvd))

        if return_courant:
            courant = courant.sort_index()
            courant.loc[csv_output_segments].to_csv(output_path.joinpath(filename_courant))

        # usgs_df_filtered = usgs_df[usgs_df.index.isin(csv_output_segments)]
        # usgs_df_filtered.to_csv(output_path.joinpath("usgs_df.csv"))

    # Write out LastObs as netcdf.
    # Assumed that this capability is only needed for AnA simulations
    # i.e. when timeslice files are provided to support DA
    data_assimilation_folder = data_assimilation_parameters.get(
    "data_assimilation_timeslices_folder", None
    )
    lastobs_output_folder = data_assimilation_parameters.get(
    "lastobs_output_folder", None
    )
    # if lastobs_output_folder:
    #     warnings.warn("No LastObs output folder directory specified in input file - not writing out LastObs data")
    if data_assimilation_folder and lastobs_output_folder:
        nhd_io.lastobs_df_output(
            lastobs_df,
            dt,
            nts,
            t0,
            link_gage_df['gages'],
            lastobs_output_folder,
        )

    if debuglevel <= -1:
        print(flowveldepth)

    if verbose:
        print("output complete")
    if showtiming:
        print("... in %s seconds." % (time.time() - start_time))

    ################### Parity Check

    if parity_set:
        if verbose:
            print(
                "conducting parity check, comparing WRF Hydro results against t-route results"
            )
        if showtiming:
            start_time = time.time()

        parity_check(
            parity_set, results,
        )

        if verbose:
            print("parity check complete")
        if showtiming:
            print("... in %s seconds." % (time.time() - start_time))
