"""top level run script"""

import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
from hdmf.common.table import DynamicTable
from hdmf_zarr.nwb import NWBZarrIO
from pynwb import NWBHDF5IO

data_folder = Path("../data/")
scratch_folder = Path("../scratch/")
results_folder = Path("../results/")


# converts channel name of the form 'LFP1' or 'AP1' to idx of the form 0
def channel_name_to_idx(channel_name):
    """
    Convert a channel name to a corresponding channel index.

    This function processes a channel name by removing specific substrings ('LFP' and 'AP')
    and then attempts to convert the resulting string into an integer. The function returns
    the integer index of the channel, adjusting by subtracting 1. If the conversion fails,
    an exception is raised indicating an unexpected channel name format.

    Parameters
    ----------
    channel_name : str
        The name of the channel as a string. The function expects the name to possibly
        include the substrings 'LFP' or 'AP', which will be removed during processing.

    Returns
    -------
    int
        The channel index, derived from the numerical part of the channel name, with
        1 subtracted from it.
    """
    stripped_name = channel_name.replace("LFP", "").replace("AP", "")
    try:
        return int(stripped_name) - 1
    except:
        raise Exception("Unexpected channel name format")


def build_ccf_map(ccf_json_files) -> dict:
    """
    Build a CCF (Common Coordinate Framework) map from a list of JSON files.

    This function processes multiple JSON files that describe the location and
    structure of brain regions for different probes. It extracts the relevant
    information from each JSON file and constructs a dictionary mapping each probe
    and channel to the corresponding brain region and coordinates (x, y, z).

    Parameters
    ----------
    ccf_json_files : list of pathlib.Path
        A list of file paths (as Path objects) to the JSON files containing CCF data.
        Each file is expected to have data for various channels, with information
        on the brain region and the corresponding coordinates.

    Returns
    -------
    dict
        A dictionary where the keys are tuples of the form `(probe_name, channel_id)`,
        and the values are lists containing the structure name and coordinates `[structure, x, y, z]`.
    """
    ccf_map = {}
    for ccf_json_path in ccf_json_files:
        print("Reading", ccf_json_path)
        probe_name = ccf_json_path.parent.stem
        probe_name = probe_name[0].lower() + probe_name[1:]
        with open(ccf_json_path, "r") as f:
            ccf_json_info = json.load(f)

        for channel in ccf_json_info:
            channel_id = channel[channel.index("_") + 1 :]
            structure = ccf_json_info[channel]["brain_region"]
            x = ccf_json_info[channel]["x"]
            y = ccf_json_info[channel]["y"]
            z = ccf_json_info[channel]["z"]

            ccf_map[probe_name, int(channel_id)] = [structure, x, y, z]

    return ccf_map


def get_new_electrode_colums(nwb, ccf_map):
    """
    Extract the anatomical location (brain region) and coordinates (x, y, z)
    for each electrode from the provided NWB file and CCF map.

    This function loops over all electrodes in the given NWB file, retrieves
    the probe name and channel ID, and looks up the corresponding brain region
    and coordinates from the CCF map. It returns the lists of brain regions and
    coordinates for the electrodes.

    Parameters
    ----------
    nwb : pynwb.file.NWBFile
        An NWB (Neurodata Without Borders) file containing electrode information
        (e.g., electrode group names and channel names).

    ccf_map : dict
        A dictionary mapping probe names and channel IDs (as tuples) to the
        corresponding brain region and coordinates. Each key is a tuple of the form
        `(probe_name, channel_id)`, and each value is a list `[structure, x, y, z]`.

    Returns
    -------
    tuple of lists
        A tuple containing four lists:
        - `locs`: A list of brain regions (structure names) corresponding to each electrode.
        - `xs`: A list of x-coordinates for each electrode.
        - `ys`: A list of y-coordinates for each electrode.
        - `zs`: A list of z-coordinates for each electrode.
    """
    locs, xs, ys, zs = [], [], [], []
    for row in nwb.electrodes:
        probe_name = row["group_name"].item()
        probe_name = probe_name.replace(" ", "")
        probe_name = probe_name[0].lower() + probe_name[1:]
        channel_id = channel_name_to_idx(row["channel_name"].item())
        try:
            structure, x, y, z = ccf_map[probe_name, channel_id]
        except KeyError:
            print(
                f"CCF information for an electrode ({probe_name}, channel {channel_id}) not found. Perhaps no output from IBL alignment for {probe_name}"
            )
            continue

        locs.append(structure)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return locs, xs, ys, zs


def run():
    ccf_json_files = tuple(data_folder.rglob("ccf*.json"))
    if not ccf_json_files:
        raise FileNotFoundError("No ibl json output attached")

    # find base NWB file
    nwb_files = [
        p
        for p in (data_folder / "nwb").iterdir()
        if p.name.endswith(".nwb") or p.name.endswith(".nwb.zarr")
    ]
    assert len(nwb_files) == 1, "Attach one base NWB file data at a time"

    source_path = nwb_files[0]
    # source_path = '/root/capsule/data/ecephys_737812_2024-07-30_16-17-43_sorted_2024-10-15_02-38-10/nwb/ecephys_737812_2024-07-30_16-17-43_experiment1_recording1.nwb'
    destination_dir = "/results/nwb"
    destination_path = os.path.join(destination_dir, os.path.basename(source_path))
    print(source_path, destination_path)
    shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
    nwbfile_input_path = Path(destination_path)

    if nwbfile_input_path.is_dir():
        assert (
            nwbfile_input_path / ".zattrs"
        ).is_file(), f"{nwbfile_input_path.name} is not a valid Zarr folder"
        NWB_BACKEND = "zarr"
        NWB_SUFFIX = ".nwb.zarr"
        io_class = NWBZarrIO
    else:
        NWB_BACKEND = "hdf5"
        NWB_SUFFIX = ".nwb"
        io_class = NWBHDF5IO
    print(f"NWB backend: {NWB_BACKEND}")

    print("Building CCF Map from IBL jsons")
    ccf_map = build_ccf_map(ccf_json_files)

    print("Reading NWB in append mode:", nwbfile_input_path)
    with io_class(str(nwbfile_input_path), mode="a") as read_io:
        nwb = read_io.read()

        print("Getting new electrode columns")
        locs, xs, ys, zs = get_new_electrode_colums(nwb, ccf_map)
        if len(locs) < len(nwb.electrodes):
            for i in range(len(nwb.electrodes) - len(locs)):
                locs.append("unknown")
                xs.append(np.nan)
                ys.append(np.nan)
                zs.append(np.nan)

        nwb.electrodes.location.data[:] = np.array(locs)
        nwb.electrodes.add_column("x", "ccf x coordinate", data=xs)
        nwb.electrodes.add_column("y", "ccf y coordinate", data=ys)
        nwb.electrodes.add_column("z", "ccf z coordinate", data=zs)

        nwbfile_output_path = results_folder / f"{nwbfile_input_path.stem}.nwb"
        print("Exporting to NWB:", nwbfile_output_path)
        with io_class(str(nwbfile_output_path), "w") as export_io:
            export_io.export(
                src_io=read_io, nwbfile=nwb, write_args={"link_data": False}
            )
        print(f"Done writing {nwbfile_output_path}")


if __name__ == "__main__":
    run()
