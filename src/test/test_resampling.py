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
        "sine_wave": "sine-2+marker.xdf"
    }.items()
    if (path / value).exists()
}
path = Path("/home/mirko/Documents/code/Neurolive/pyxdf_venv/src/pdxdf/src/test")
dfs = {
    key: np.genfromtxt(path / value, delimiter=',', dtype="float32")  
    for key, value in {
        "id2_time_series": "sine_wave_id2_time_series.csv",
        "id2_time_stamps": "sine_wave_id2_time_stamps.csv"
        # "another_time_series": "another_time_series.csv"
    }.items()
    if (path / value).exists()
}

# open file and visualize the data
streams, header = load_xdf(files["sine_wave"])
print(header["info"]["version"][0] )

for stream in streams:
    y = stream["time_series"]
    print("====================================")
    print("name: ", stream["info"]["name"][0])

    if stream["info"]["name"][0] == "Test data stream 0 counter sine+":
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
        print("clock_offsets: ", stream["footer"]["info"]["clock_offsets"][0]["offset"][0])

        # pprint.pprint(stream, width=120, compact=True)
        print("CLOCK TIMES")
        print(stream["clock_times"])
        print("CLOCK VALUES")   
        formated_clock_values = [format(float(value), '.8f') for value in stream["clock_values"]]
        print(stream["clock_values"])
        print("TIME SERIES")
        pprint.pprint(stream["time_series"])
        output_path = path / "sine_wave_id2_time_series.csv"
        np.savetxt(output_path, stream["time_series"], delimiter=",", fmt='%s')
        print("TIME STAMPS")
        pprint.pprint(stream["time_stamps"])
        output_path = path / "sine_wave_id2_time_stamps.csv"
        np.savetxt(output_path, stream["time_stamps"], delimiter=",", fmt='%s')

        # print("CLOCK OFFESTS")
        # for clock_offset in stream["footer"]["info"]["clock_offsets"]:
        #     print("TIME:")
        #     for time in clock_offset["offset"]:
        #         print(time["time"][0], ",")
        #     print("VALUE:")
        #     for value in clock_offset["offset"]:
        #         #print(format(float(value["value"][0]), '.8f'), ",")
        #         print(value["value"][0], ",")

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
@pytest.mark.skipif("sine_wave" not in files, reason="Test file not found")
def test_sine_wave_file(synchronize_clocks):
    path = files["sine_wave"]   
    streams, header = load_xdf(
        path, 
        synchronize_clocks=synchronize_clocks,
    )
    assert header["info"]["version"][0] == "1.0"

    #================================================================================================
    # Stream ID: ctrl
    #================================================================================================
    i = 0
    assert len(streams) == 4
    assert streams[i]["info"]["name"][0] == "ctrl"
    assert streams[i]["info"]["type"][0] == "control"
    assert streams[i]["info"]["channel_count"][0] == "1"
    assert streams[i]["info"]["nominal_srate"][0] == "0.000000000000000"
    assert streams[i]["info"]["channel_format"][0] == "string"
    assert streams[i]["info"]["created_at"][0] == "15415.43635413100"
    desc = streams[i]["info"]["desc"][0]
    assert isinstance(desc, dict), f"Stream {i} desc not a dict {type(desc)}"
    assert streams[i]["info"]["uid"][0] == "3fefaf0e-4732-4a6d-a31d-a4add58890fd"

    # Info added by pyxdf
    assert streams[i]["info"]["stream_id"] == 2
    assert streams[i]["info"]["effective_srate"] == 0
    assert streams[i]["info"]["segments"] == [(0, 0)]
    assert streams[i]["info"]["clock_segments"] == (
        [(0, 0)] if synchronize_clocks else []
    )

    # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "15468.027867873"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "15468.027867873"
    assert streams[i]["footer"]["info"]["sample_count"][0] == "1"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "15466.657564722"
    assert first_clock_offset["value"][0] == "-2.568699983385159e-05"

    # Check Clock offset
    hc_time = np.array(
        [
            15466.657564722,
            15471.657667209,
            15476.657783314,
            15481.6579105335,
            15486.6580300445,
            15491.6581467995,
            15496.658254298,
            15501.658366477,
            15506.6584887995,
            15511.658595879,
            15516.658739848,
            15521.658826143,
            15526.6588890175,
            15531.6591194665,
            15536.659345346,
            15541.659592276,
            15546.6597899145,
            15551.6599490125,
            15556.660155582,
            15561.6603913755,
            15566.660553395,
            15571.6608108525,
            15576.661004165,
            15581.6612651325,
        ],
        dtype=np.float64
    )
    time = np.array([],dtype=np.float64)
    for clock_offset_time in streams[i]["footer"]["info"]["clock_offsets"][0]["offset"]:
        time = np.append(time, float(clock_offset_time["time"][0]))
    np.testing.assert_equal(time, hc_time)

    hc_value = np.array(
            [
                -2.568699983385159e-05,
                -1.009799962048419e-05,
                -1.556600000185426e-05,
                -2.777850022539496e-05,
                -1.613649965293007e-05,
                -6.199500603543129e-06,
                -1.321899981121533e-05,
                -1.668099957896629e-05,
                -1.390350007568486e-05,
                -3.199000275344588e-06,
                -1.221700040332507e-05,
                -6.5909998738789e-06,
                -1.049499587679747e-06,
                -2.562650024628965e-05,
                -6.638999911956489e-05,
                -6.739600030414294e-05,
                -5.483750010171207e-05,
                -1.001249984255992e-05,
                -9.386999408889096e-06,
                -3.934750020562205e-05,
                3.238999852328561e-06,
                -6.74225002512685e-05,
                -1.123200036090566e-05,
                -3.099150035268394e-05,
            ],
            dtype=np.float64
        )
    value = np.array([],dtype=np.float64)
    for clock_offset_value in streams[i]["footer"]["info"]["clock_offsets"][0]["offset"]:
        value = np.append(value, float(clock_offset_value["value"][0]))
        print(value)
    np.testing.assert_equal(value, hc_value)

    hc_clock_time = [
        15466.657564722002,
        15471.657667209,
        15476.657783314,
        15481.6579105335,
        15486.658030044498,
        15491.658146799502,
        15496.658254298,
        15501.658366477,
        15506.6584887995,
        15511.658595879,
        15516.658739848,
        15521.658826143,
        15526.6588890175,
        15531.659119466502,
        15536.659345346,
        15541.659592276,
        15546.659789914502,
        15551.659949012499,
        15556.660155581998,
        15561.6603913755,
        15566.660553395,
        15571.6608108525,
        15576.661004165002,
        15581.661265132501
    ]
    np.testing.assert_equal(streams[i]["clock_times"], hc_clock_time)

    #================================================================================================
    # Stream ID: Test data stream 0 counter
    #================================================================================================
    
    i = 1
    assert streams[i]["info"]["name"][0] == "Test marker stream 0 counter"
    assert streams[i]["info"]["type"][0] == "marker"
    assert streams[i]["info"]["channel_count"][0] == '1'
    assert streams[i]["info"]["nominal_srate"][0] == '0.000000000000000'
    assert streams[i]["info"]["channel_format"][0] == "string"
    assert streams[i]["info"]["created_at"][0] == "15440.76695831100"
    desc = streams[i]["info"]["desc"][0]
    assert isinstance(desc, dict), f"Stream {i} desc not a dict {type(desc)}"
    assert streams[i]["info"]["uid"][0] == "79ec545f-3aa4-42e1-b158-eb01e4a1a911"

    # Info added by pyxdf
    assert streams[i]["info"]["stream_id"] == 3
    assert streams[i]["info"]["effective_srate"] == 0
    assert streams[i]["info"]["segments"] == [(0, 61)]
    assert streams[i]["info"]["clock_segments"] == (
        [(0, 61)] if synchronize_clocks else []
    )

    # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "15468.02782607"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "15528.028417884"
    assert streams[i]["footer"]["info"]["sample_count"][0] == "62"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "15466.657493655"
    first_clock_offest_value = format(float(first_clock_offset["value"][0]), '.8f')
    assert  first_clock_offest_value == "0.00000757"

    # Time-series data
    s = np.array(
        [
            ['0'], ['1'], ['2'], ['3'], ['4'], ['5'], ['6'], ['7'], ['8'], ['9'], ['10'], ['11'], ['12'], ['13'], ['14'], ['15'], ['16'], ['17'], ['18'], ['19'], ['20'], ['21'], ['22'], ['23'], ['24'], ['25'], ['26'], ['27'], ['28'], ['29'], ['30'], ['31'], ['32'], ['33'], ['34'], ['35'], ['36'], ['37'], ['38'], ['39'], ['40'], ['41'], ['42'], ['43'], ['44'], ['45'], ['46'], ['47'], ['48'], ['49'], ['50'], ['51'], ['52'], ['53'], ['54'], ['55'], ['56'], ['57'], ['58'], ['59'], ['60'], ['']
        ],
        dtype=np.object_
    )

    t = np.array(
        [15468.02781961, 15469.02781954, 15470.02781947, 15471.0278194 ,
       15472.02781934, 15473.02781927, 15474.0278192 , 15475.02781914,
       15476.02781907, 15477.027819  , 15478.02781894, 15479.02781887,
       15480.0278188 , 15481.02781874, 15482.02781867, 15483.0278186 ,
       15484.02781853, 15485.02781847, 15486.0278184 , 15487.02781833,
       15488.02781827, 15489.0278182 , 15490.02781813, 15491.02781807,
       15492.027818  , 15493.02781793, 15494.02781786, 15495.0278178 ,
       15496.02781773, 15497.02781766, 15498.0278176 , 15499.02781753,
       15500.02781746, 15501.0278174 , 15502.02781733, 15503.02781726,
       15504.0278172 , 15505.02781713, 15506.02781706, 15507.02781699,
       15508.02781693, 15509.02781686, 15510.02781679, 15511.02781673,
       15512.02781666, 15513.02781659, 15514.02781653, 15515.02781646,
       15516.02781639, 15517.02781632, 15518.02781626, 15519.02781619,
       15520.02781612, 15521.02781606, 15522.02781599, 15523.02781592,
       15524.02781586, 15525.02781579, 15526.02781572, 15527.02781566,
       15528.02781559, 15528.0284074 ],
        dtype=np.float64
    )

    # if synchronize_clocks:
    #     t = t -first_clock_offest_value

    np.testing.assert_equal(streams[i]["time_series"], s)
    np.testing.assert_allclose(streams[i]["time_stamps"], t)

    # Clock Offsets
    clock_times = np.asarray(
        [15466.657493655, 15471.6576702, 15476.6577768835, 15481.657888327001, 15486.658017618502, 15491.6581525625, 15496.658256522001, 15501.658357677501, 15506.658488678499, 15511.658597907, 15516.6587569085, 15521.65893407, 15526.659000642501, 15531.6591057945, 15536.6592940335, 15541.659522168502, 15546.6597340835, 15551.659938088502, 15556.6601353435, 15561.660350460501, 15566.660569894499, 15571.660755384499, 15576.660994366499, 15581.661233665502]
    )
    
    clock_values = np.asarray(
        [7.571999958599918e-06, -1.375600004394073e-05, -4.471499778446741e-06, -5.567000698647462e-06, -3.7045001590740867e-06, -1.1959500625380315e-05, -1.5439000890182797e-05, -7.877500138420146e-06, -1.0744500286818948e-05, -5.218000296736136e-06, -2.9280499802553095e-05, -1.0330999430152588e-05, -2.814500476233661e-06, -1.1977500435023103e-05, -1.1977500435023103e-05, -1.1977500435023103e-05, -1.1977500435023103e-05, -1.1977500435023103e-05, -1.1977500435023103e-05, -1.1977500435023103e-05, -1.1977500435023103e-05, -1.1977500435023103e-05, -1.1977500435023103e-05, -1.1977500435023103e-05]
    )
    np.testing.assert_equal(streams[i]["clock_times"], clock_times)
    np.testing.assert_equal(streams[i]["clock_values"], clock_values)

    #================================================================================================
    # Stream ID: Test data stream 0 counter sine+
    #================================================================================================
    i = 2
    assert streams[i]["info"]["name"][0] == "Test data stream 0 counter sine+"
    assert streams[i]["info"]["type"][0] == "eeg"
    assert streams[i]["info"]["channel_count"][0] == '8'
    assert streams[i]["info"]["nominal_srate"][0] == '512.0000000000000'
    assert streams[i]["info"]["channel_format"][0] == "float32"
    assert streams[i]["info"]["created_at"][0] == '15440.81677496200'
    desc = streams[i]["info"]["desc"][0]
    assert isinstance(desc, dict), f"Stream {i} desc not a dict {type(desc)}"
    assert streams[i]["info"]["uid"][0] == "3c30408c-1561-405e-9db3-bdd438659219"

    # Info added by pyxdf
    assert streams[i]["info"]["stream_id"] == 1
    assert streams[i]["info"]["effective_srate"] ==  512.0000186214386 if synchronize_clocks else 512.0
    assert streams[i]["info"]["segments"] == [(0, 30720), (30721, 30721)]
    assert streams[i]["info"]["clock_segments"] == (
        [(0, 30721)] if synchronize_clocks else []
    )

    # Footer
    assert streams[i]["footer"]["info"]["first_timestamp"][0] == "15468.02782607"
    assert streams[i]["footer"]["info"]["last_timestamp"][0] == "15527.030084352"
    assert streams[i]["footer"]["info"]["sample_count"][0] == "30722"
    first_clock_offset = streams[i]["footer"]["info"]["clock_offsets"][0]["offset"][0]
    assert first_clock_offset["time"][0] == "15466.657574327"
    first_clock_offest_value = format(float(first_clock_offset["value"][0]), '.8f')
    assert  first_clock_offest_value == "-0.00000106"
    
    # Time-series data
    assert np.array_equal(streams[i]["time_series"], dfs["id2_time_series"])
    np.testing.assert_allclose(streams[i]["time_stamps"], dfs["id2_time_stamps"])

    # Clock Offsets
    assert streams[i]["clock_times"] == pytest.approx(
        float(first_clock_offset["time"][0]),abs=1e-6
    )
    assert streams[i]["clock_values"] == pytest.approx(
        float(first_clock_offset["value"][0]),abs=1e-6
    )
    