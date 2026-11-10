"""Raw CALCE XLSX parsing integration tests (no processed CSVs)."""
from __future__ import annotations

import io
import zipfile

import numpy as np

from paper_reproduction.data import _calce_workbook_cycles


def make_source_workbook() -> bytes:
    names = [
        "Data_Point", "Test_Time(s)", "Cycle_Index",
        "Current(A)", "Voltage(V)", "Discharge_Capacity(Ah)",
    ]
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    spreadsheet = ['<?xml version="1.0" encoding="UTF-8"?>',
                   f'<worksheet xmlns="{ns}"><sheetData>']
    letters = "ABCDEF"
    head = "".join(
        f'<c r="{letters[i]}1" t="inlineStr"><is><t>{name}</t></is></c>'
        for i, name in enumerate(names)
    )
    spreadsheet.append(f'<row r="1">{head}</row>')
    for i in range(12):
        # Two independent synthetic cycles, six voltage/current samples each.
        cycle = 1 if i < 6 else 2
        k = i % 6
        values = [i + 1, k * 30.0, cycle, -1.0, 3.9-0.04*k, .1*k]
        cells = "".join(
            f'<c r="{letters[j]}{i+2}"><v>{v}</v></c>'
            for j, v in enumerate(values)
        )
        spreadsheet.append(f'<row r="{i+2}">{cells}</row>')
    spreadsheet.append("</sheetData></worksheet>")
    workbook = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Channel_1-011" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as z:
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", rels)
        z.writestr("xl/worksheets/sheet1.xml", "".join(spreadsheet))
    return output.getvalue()


def test_raw_calce_channel_sheet_read_and_capacity_without_summary_csv():
    workbook = make_source_workbook()
    records = list(_calce_workbook_cycles(workbook, "test_calce.xlsx", 32))
    assert len(records) == 2
    for idx, cap, trace in records:
        assert idx in (1, 2)
        assert abs(cap-.5) < 1e-7
        assert trace.shape == (32, 4)
        np.testing.assert_allclose(trace[:, 2], 0)
        assert trace[0, 0] > trace[-1, 0]  # Voltage decreases
        np.testing.assert_allclose(trace[0, 3], 0)
        np.testing.assert_allclose(trace[-1, 3], 150)
