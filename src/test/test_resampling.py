"""
- Timestamps align when synchronize_clocks=True/False.
- original vs. resampled data 
- resampled timestamps are evenly spaced.
- data structure maintained after resampling.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pytest
from pyxdf import load_xdf
import pprint
import pandas as pd

path = Path("/home/mirko/Documents/code/Neurolive/example-files")

files = {
    key: path / value
    for key, value in {
        "resampling": "resampling.xdf"
    }.items()
    if (path / value).exists()
}
path = Path("/home/mirko/Documents/code/Neurolive/pyxdf_venv/src/pdxdf/src/test")
dfs = {
    key: np.genfromtxt(path / value, delimiter=',', dtype="float64")  
    for key, value in {
        "id1_time_series": "resample_stream_id1_time_series.csv",
        "id1_time_stamps": "resample_stream_id1_time_stamps.csv",
        "id3_time_series": "resample_stream_id3_time_series.csv",
        "id3_time_stamps": "resample_stream_id3_time_stamps.csv",
    }.items()
    if (path / value).exists()
}

# open file and visualize the data
streams, header = load_xdf(files["resampling"])
print(header["info"]["version"][0] )

for stream in streams:
    y = stream["time_series"]
    print("====================================")
    print("name: ", stream["info"]["name"][0])

    if stream["info"]["name"][0] == "Resample: Test data stream 1":
        print("type: ", stream["info"]["type"][0])
        print("channel_count: ",stream["info"]["channel_count"][0])
        print("nominal_srate: ", stream["info"]["nominal_srate"][0])
        print("channel_format: ", stream["info"]["channel_format"][0])
        print("created_at: ", stream["info"]["created_at"][0])
        print("desc: ", stream["info"]["desc"][0])
        print("uid: ", stream["info"]["uid"][0])    
        print("PYXDF INFO")
        print("stream_id: ", stream["info"]["stream_id"])
        print("effective_srate: ", stream["info"]["effective_srate"])
        print("segments: ", stream["info"]["segments"])
        print("clock_segments: ", stream["info"]["clock_segments"])
        print("FOOTER INFO")
        print("first_timestamp: ", stream["footer"]["info"]["first_timestamp"][0])
        print("last_timestamp: ", stream["footer"]["info"]["last_timestamp"][0])
        print("sample_count: ", stream["footer"]["info"]["sample_count"][0])
        raw_clock_offset = stream["footer"]["info"]["clock_offsets"][0]["offset"][0]
        formated_clock_offset_time = format(float(raw_clock_offset["time"][0]), '.8f')
        print("CLOCK OFFSET")
        print("time: ", raw_clock_offset["time"][0], "formated:",formated_clock_offset_time)
        formated_clock_offset_value = format(float(raw_clock_offset["value"][0]), '.8f')
        print("value: ", raw_clock_offset["value"][0], "formated:", formated_clock_offset_value)    
        # pprint.pprint(stream, width=120, compact=True)
        print("CLOCK TIMES")
        print(stream["clock_times"])
        print("CLOCK VALUES")   
        formated_clock_values = [format(float(value), '.8f') for value in stream["clock_values"]]
        print(stream["clock_values"])
        # print("TIME SERIES")
        pprint.pprint(stream["time_series"])
        output_path = path / f'resample_stream_id{stream["info"]["stream_id"]}_time_series.csv'
        np.savetxt(output_path, stream["time_series"], delimiter=",", fmt='%s')

        #Check  data integrity
        print("TIME SERIES SHAPES")
        print("stream:", stream["time_series"].shape)
        print("dfs:", dfs["id1_time_series"].shape)

        print("TIME SERIES TYPES")
        print("stream", stream["time_series"].dtype)
        print("dfs", dfs["id1_time_series"].dtype)


        print("TIME STAMPS")
        pprint.pprint(stream["time_stamps"])
        output_path = path / f'resample_stream_id{stream["info"]["stream_id"]}_time_stamps.csv'
        np.savetxt(output_path, stream["time_stamps"], delimiter=",", fmt='%s')
        
        #Check  data integrity
        print("TIME STAMPS SHAPES")
        print("stream:", stream["time_stamps"].shape)
        print("dfs:", dfs["id1_time_stamps"].shape)

        print("TIME SERIES TYPES")
        print("stream", stream["time_stamps"].dtype)
        print("dfs", dfs["id1_time_stamps"].dtype)

        print("CLOCK OFFESTS")
        for clock_offset in stream["footer"]["info"]["clock_offsets"]:
            print("TIME:")
            for time in clock_offset["offset"]:
                print(time["time"][0], ",")
            print("VALUE:")
            for value in clock_offset["offset"]:
                #print(format(float(value["value"][0]), '.8f'), ",")
                print(value["value"][0], ",")

    if isinstance(y, list):
        # list of strings, draw one vertical line for each marker
        for timestamp, marker in zip(stream["time_stamps"], y):
            plt.axvline(x=timestamp)
            # print(f'Marker "{marker[0]}" @ {timestamp:.2f}s')
    elif isinstance(y, np.ndarray):
        # numeric data, draw as lines
        # plt.plot(stream["time_stamps"], y)
        print("plotting")
    else:
        raise RuntimeError("Unknown stream format")

# plt.show()


@pytest.mark.parametrize("synchronize_clocks", [False, True])
@pytest.mark.skipif("resampling" not in files, reason="Test file not found")
def test_sine_wave_file(synchronize_clocks):
    path = files["resampling"]   
    streams, header = load_xdf(
        path, 
        synchronize_clocks=synchronize_clocks,
    )
    assert header["info"]["version"][0] == "1.0"

    #================================================================================================
    # Stream ID: Resample: Test marker stream 0
    #================================================================================================
    i = 1
    assert streams[i]["info"]["name"][0] == "Resample: Test marker stream 0"
    assert streams[i]["info"]["type"][0] == "marker"
    assert streams[i]["info"]["channel_count"][0] == "1"
    assert streams[i]["info"]["nominal_srate"][0] == "0.000000000000000"
    assert streams[i]["info"]["channel_format"][0] == "string"
    assert streams[i]["info"]["created_at"][0] == "80693.65198860300"
    desc = streams[i]["info"]["desc"][0]
    assert isinstance(desc, dict), f"Stream {i} desc not a dict {type(desc)}"
    assert streams[i]["info"]["uid"][0] == "f380306e-ea4c-42f1-af9f-34143e238d43"

    # Info added by pyxdf
    assert streams[i]["info"]["stream_id"] == 2
    assert streams[i]["info"]["effective_srate"] == 0
    assert streams[i]["info"]["segments"] == [(0, 60)]
    assert streams[i]["info"]["clock_segments"] == (
        [(0, 60)] if synchronize_clocks else []
    )

    # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "80754.15117545699"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "80814.15117545699"
    assert streams[i]["footer"]["info"]["sample_count"][0] == "61"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "80715.10266161899"
    assert first_clock_offset["value"][0] == "-1.568199513712898e-05"

    # Check Clock offset
    hc_time = np.array(
        [80715.10266161899, 80720.102860616, 80725.103058195, 80730.1032885235, 80735.103389942, 80740.1036506505, 80745.1039255875, 80750.1041835065, 80755.10426025299, 80760.104329771, 80765.10446785999, 80770.10466277, 80775.1047253935, 80780.10480736449, 80785.104876335, 80790.10496181651, 80795.105037141, 80800.10519147202, 80805.1052793545, 80810.10538661151, 80815.10558957249, 80820.10585804548, 80825.10609295151],
        dtype=np.float64
    )
    time = np.array([],dtype=np.float64)
    for clock_offset_time in streams[i]["footer"]["info"]["clock_offsets"][0]["offset"]:
        time = np.append(time, float(clock_offset_time["time"][0]))
    np.testing.assert_equal(time, hc_time)

    hc_value = np.array(
        [-1.568199513712898e-05, -1.532099122414365e-05, -1.285199687117711e-05, -2.394449984421954e-05, -2.85299975075759e-05, -2.762550138868392e-05, -3.014149842783809e-05, -2.776649489533156e-05, 9.6160001703538e-06, 1.934499596245587e-05, -3.109998942818493e-06, -1.66569952853024e-05, -8.77749698702246e-06, -1.09725006041117e-05, -9.801995474845171e-06, -1.091502781491727e-06, -6.891001248732209e-06, -7.264003215823323e-06, -1.459506165701896e-06, -5.729503754992038e-06, -1.241149584529921e-05, -2.565449540270492e-05, -2.382250386290252e-05],
        dtype=np.float64
        )
    
    value = np.array([],dtype=np.float64)
    for clock_offset_value in streams[i]["footer"]["info"]["clock_offsets"][0]["offset"]:
        value = np.append(value, float(clock_offset_value["value"][0]))
    np.testing.assert_equal(value, hc_value)

    hc_clock_time = [80715.10266161899, 80720.102860616, 80725.103058195, 80730.1032885235, 80735.103389942, 80740.1036506505, 80745.1039255875, 80750.1041835065, 80755.10426025299, 80760.104329771, 80765.10446785999, 80770.10466277, 80775.1047253935, 80780.10480736449, 80785.104876335, 80790.10496181651, 80795.105037141, 80800.10519147202, 80805.1052793545, 80810.10538661151, 80815.10558957249, 80820.10585804548, 80825.10609295151]
    np.testing.assert_equal(streams[i]["clock_times"], hc_clock_time)


    #================================================================================================
    # Stream ID: Resample: Test marker stream 1
    #================================================================================================
    
    i = 0
    assert streams[i]["info"]["name"][0] == "Resample: Test marker stream 1"
    assert streams[i]["info"]["type"][0] == "marker"
    assert streams[i]["info"]["channel_count"][0] == '1'
    assert streams[i]["info"]["nominal_srate"][0] == '0.000000000000000'
    assert streams[i]["info"]["channel_format"][0] == "string"
    assert streams[i]["info"]["created_at"][0] == "80693.65221820401"
    desc = streams[i]["info"]["desc"][0]
    assert isinstance(desc, dict), f"Stream {i} desc not a dict {type(desc)}"
    assert streams[i]["info"]["uid"][0] == "8d489ab3-8435-438c-b6bf-48c9abc25c62"

    # # Info added by pyxdf
    assert streams[i]["info"]["stream_id"] == 4
    assert streams[i]["info"]["effective_srate"] == 0
    assert streams[i]["info"]["segments"] == [(0, 60)]
    assert streams[i]["info"]["clock_segments"] == (
        [(0, 60)] if synchronize_clocks else []
    )

    # # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "80754.15117545699"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "80814.15117545699"
    assert streams[i]["footer"]["info"]["sample_count"][0] == "61"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "80715.10254538601"
    first_clock_offest_value = format(float(first_clock_offset["value"][0]), '.8f')
    assert  first_clock_offest_value == "-0.00003145"

    # Time-series data
    s = np.array(
        [['0'], ['1'], ['2'], ['3'], ['4'], ['5'], ['6'], ['7'], ['8'], ['9'], ['10'], ['11'], ['12'], ['13'], ['14'], ['15'], ['16'], ['17'], ['18'], ['19'], ['20'], ['21'], ['22'], ['23'], ['24'], ['25'], ['26'], ['27'], ['28'], ['29'], ['30'], ['31'], ['32'], ['33'], ['34'], ['35'], ['36'], ['37'], ['38'], ['39'], ['40'], ['41'], ['42'], ['43'], ['44'], ['45'], ['46'], ['47'], ['48'], ['49'], ['50'], ['51'], ['52'], ['53'], ['54'], ['55'], ['56'], ['57'], ['58'], ['59'], ['60']],
        dtype=np.object_
    )

    t = np.array(
        [80754.15115301, 80755.15115322, 80756.15115342, 80757.15115363,
        80758.15115384, 80759.15115404, 80760.15115425, 80761.15115446,
        80762.15115466, 80763.15115487, 80764.15115508, 80765.15115528,
        80766.15115549, 80767.1511557 , 80768.1511559 , 80769.15115611,
        80770.15115632, 80771.15115652, 80772.15115673, 80773.15115694,
        80774.15115714, 80775.15115735, 80776.15115756, 80777.15115776,
        80778.15115797, 80779.15115818, 80780.15115838, 80781.15115859,
        80782.1511588 , 80783.151159  , 80784.15115921, 80785.15115942,
        80786.15115962, 80787.15115983, 80788.15116004, 80789.15116024,
        80790.15116045, 80791.15116066, 80792.15116086, 80793.15116107,
        80794.15116128, 80795.15116148, 80796.15116169, 80797.1511619,         80798.1511621 , 80799.15116231, 80800.15116252, 80801.15116272,
        80802.15116293, 80803.15116314, 80804.15116334, 80805.15116355,
        80806.15116376, 80807.15116396, 80808.15116417, 80809.15116438,
        80810.15116458, 80811.15116479, 80812.151165  , 80813.1511652, 
        80814.15116541],
        dtype=np.float64
    )

    # if synchronize_clocks:
    #     t = t -first_clock_offest_value

    np.testing.assert_equal(streams[i]["time_series"], s)
    np.testing.assert_allclose(streams[i]["time_stamps"], t)

    # Clock Offsets
    clock_times = np.asarray(
        [80715.102545386, 80720.10276420251, 80725.1030050005, 80730.103241226, 80735.10347608049, 80740.10368796551, 80745.10390989101, 80750.10418351149, 80755.104318855, 80760.104343779, 80765.10443810749, 80770.1045385995, 80775.1046277535, 80780.10470479901, 80785.1047848265, 80790.104881465, 80795.10498196099, 80800.105102966, 80805.10521586199, 80810.1053525545, 80815.105591623, 80820.10582849401, 80825.106114797]    )
    
    clock_values = np.asarray(
        [-3.144900256302208e-05, -3.169850242557004e-05, -3.087049844907597e-05, -2.7586000214796513e-05, -2.8724491130560637e-05, -2.6757501473184675e-05, -1.4404002286028117e-05, -2.7716501790564507e-05, -6.435500108636916e-05, -7.716000254731625e-06, -1.1430493032094091e-05, -9.064497135113925e-06, -1.405749935656786e-05, -1.2762007827404886e-05, -9.284500265493989e-06, -1.3076998584438115e-05, 4.56900306744501e-06, -1.1797994375228882e-05, -9.125993528869003e-06, -7.020498742349446e-06, -1.4431003364734352e-05, 3.943998308386654e-06, -4.56389898317866e-05]    )
    np.testing.assert_equal(streams[i]["clock_times"], clock_times)
    np.testing.assert_equal(streams[i]["clock_values"], clock_values)

    #================================================================================================
    # Stream ID: Resample: Test data stream 0
    #================================================================================================
    i = 3
    assert streams[i]["info"]["name"][0] == "Resample: Test data stream 0"
    assert streams[i]["info"]["type"][0] == "eeg"
    assert streams[i]["info"]["channel_count"][0] == '8'
    assert streams[i]["info"]["nominal_srate"][0] == '512.0000000000000'
    assert streams[i]["info"]["channel_format"][0] == "double64"
    assert streams[i]["info"]["created_at"][0] == '80693.65084625500'
    desc = streams[i]["info"]["desc"][0]
    assert isinstance(desc, dict), f"Stream {i} desc not a dict {type(desc)}"
    assert streams[i]["info"]["uid"][0] == "08b8c9ed-2ff4-4147-a451-62027e8d43d3"

    # Info added by pyxdf
    assert streams[i]["info"]["stream_id"] == 1
    assert streams[i]["info"]["effective_srate"] == 511.9999015598437 if synchronize_clocks else 512.0
    assert streams[i]["info"]["segments"] == [(0, 30720)]
    assert streams[i]["info"]["clock_segments"] == (
        [(0, 30720)] if synchronize_clocks else []
    )

    # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "80754.15117545699"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "80814.15117545699"
    assert streams[i]["footer"]["info"]["sample_count"][0] == "30721"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "80715.102474946"
    first_clock_offest_value = format(float(first_clock_offset["value"][0]), '.8f')
    assert  first_clock_offest_value == "-0.00003235"
    
    # # Time-series data
    assert np.array_equal(streams[i]["time_series"], dfs["id1_time_series"])
    np.testing.assert_allclose(streams[i]["time_stamps"], dfs["id1_time_stamps"])

    # Clock Offsets
    clock_times = np.asarray(
        [80715.102474946, 80720.10274059999, 80725.1029702175, 80730.1032226095, 80735.10336406701, 80740.10364178701, 80745.103915976, 80750.10417866449, 80755.10427381052, 80760.10434253199, 80765.10443596449, 80770.104553903, 80775.1046297205, 80780.10469649499, 80785.104780122, 80790.10488330149, 80795.10500138649, 80800.10510091999, 80805.10520407, 80810.10534903, 80815.1055766705, 80820.10582798149, 80825.106114817 ,]
    )    
    clock_values = np.asarray(
        [-3.235399344703183e-05, -2.623999898787588e-05, -1.629650068935007e-05, -2.212150138802826e-05, -2.902599953813478e-05, -2.799800131469965e-05, -2.503799623809755e-05, -2.895749639719725e-05, -1.921150396810845e-05, -4.342997272033244e-06, -6.33449963061139e-06, -1.171500480268151e-05, -8.272501872852445e-06, 7.72000930737704e-07, -3.958004526793957e-06, -2.083499566651881e-06, -1.485349639551714e-05, -9.747993317432702e-06, -1.916007022373378e-06, -3.481000021565706e-06, 5.304973456077278e-07, 4.44550096290186e-06, -4.565099516185001e-05 ]
    )

    np.testing.assert_equal(streams[i]["clock_times"], clock_times)
    np.testing.assert_allclose(streams[i]["clock_values"], clock_values)

    #================================================================================================
    # Stream ID: Resample: Test data stream 1
    #================================================================================================
    i = 2
    assert streams[i]["info"]["name"][0] == "Resample: Test data stream 1"
    assert streams[i]["info"]["type"][0] == "eeg"
    assert streams[i]["info"]["channel_count"][0] == '8'
    assert streams[i]["info"]["nominal_srate"][0] == '512.0000000000000'
    assert streams[i]["info"]["channel_format"][0] == "double64"
    assert streams[i]["info"]["created_at"][0] == '80693.65176273401'
    desc = streams[i]["info"]["desc"][0]
    assert isinstance(desc, dict), f"Stream {i} desc not a dict {type(desc)}"
    assert streams[i]["info"]["uid"][0] == "3e1ef77a-8c5b-43d5-830d-4b62b74fe233"

    # Info added by pyxdf
    assert streams[i]["info"]["stream_id"] == 3
    assert streams[i]["info"]["effective_srate"] == 511.9999042066614 if synchronize_clocks else 512.0
    assert streams[i]["info"]["segments"] == [(0, 30720)]
    assert streams[i]["info"]["clock_segments"] == (
        [(0, 30720)] if synchronize_clocks else []
    )

    # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "80754.15117545699"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "80814.15117545699"
    assert streams[i]["footer"]["info"]["sample_count"][0] == "30721"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "80715.10248696551"
    first_clock_offest_value = format(float(first_clock_offset["value"][0]), '.8f')
    assert  first_clock_offest_value == "-0.00002788"
    
    # Time-series data
    assert np.array_equal(streams[i]["time_series"], dfs["id3_time_series"])
    np.testing.assert_allclose(streams[i]["time_stamps"], dfs["id3_time_stamps"])

    # Clock Offsets
    clock_times = np.asarray(
        [80715.10248696551, 80720.1027641915, 80725.10300785449, 80730.1032387865, 80735.1034460135, 80740.10364956901, 80745.10392541249, 80750.1041842385, 80755.10433554201, 80760.10440540299, 80765.104479596, 80770.10461498398, 80775.10468138801, 80780.10476063049, 80785.10482378252, 80790.10489838099, 80795.1049939425, 80800.10512285901, 80805.10522126651, 80810.105332983, 80815.1055622585, 80820.105831665, 80825.1060673295]
    )    
    clock_values = np.asarray(
        [-2.788349956972525e-05, -3.165750240441412e-05, -3.374749940121546e-05, -2.51655001193285e-05, -2.850750024663284e-05, -2.657000732142478e-05, -2.98884988296777e-05, -2.846949792001396e-05, -3.590199776226655e-05, -3.877998096868396e-06, -8.51000368129462e-06, -6.174996087793261e-06, -1.449006958864629e-06, -9.62250487646088e-06, -5.035035428591073e-07, -4.772002284880728e-06, -7.484501111321151e-06, -2.103400038322434e-05, -1.200150290969759e-05, -9.609997505322099e-06, -4.899498890154064e-06, -2.561300061643124e-05, -2.380350633757189e-05]
    )

    np.testing.assert_equal(streams[i]["clock_times"], clock_times)
    np.testing.assert_allclose(streams[i]["clock_values"], clock_values)


@pytest.mark.parametrize("jitter_break_threshold_seconds", [0.11, 0.09])
@pytest.mark.skipif("resampling" not in files, reason="File not found.")
def test_resampling_file_segments(jitter_break_threshold_seconds):
    path = files["resampling"]
    streams, header = load_xdf(
        path,
        synchronize_clocks=True,
        jitter_break_threshold_seconds=jitter_break_threshold_seconds,
        jitter_break_threshold_samples=0,
    )

    for stream in streams: 
        nominal_srate = float(stream["info"]["nominal_srate"][0])
        if nominal_srate == 0.0:
            continue 
        tdiff = 1 / float(stream["info"]["nominal_srate"][0])
        if jitter_break_threshold_seconds > tdiff:
            assert stream["info"]["segments"] == [(0, 30720)]
            assert stream["info"]["effective_srate"] == pytest.approx(511.9999042066614)
        else:
            # Pathological case where every sample is a segment
            assert stream["info"]["segments"] == [
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
            ]
            assert stream["info"]["effective_srate"] == pytest.approx(0)
