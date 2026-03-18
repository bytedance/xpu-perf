# Copyright 2023 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fpdf import FPDF
import json
import math
import os


class PDF(FPDF):
    def titles(self, title, backend):
        self.set_xy(0.0, 0.0)
        self.set_font('Times', 'B', 16)
        # self.set_text_color(220, 50, 50)
        self.cell(w=210.0,
                  h=40.0,
                  align='C',
                  txt=title + ' REPORT (' + backend + ')',
                  border=0)

    def lines(self):
        self.rect(5.0, 5.0, 200.0, 287.0)

    def icon(self, icon_path):
        self.set_xy(10.0, 10.0)
        self.image(icon_path, link='', type='', w=37.6, h=5.2)
        self.set_xy(157.0, 0.0)
        self.set_font('Times', 'B', 10)
        # self.set_text_color(220, 50, 50)
        self.cell(w=60.0, h=25.0, align='C', txt='BYTE MLPERF', border=0)

    def charts(self, chart_path):
        self.y += 5
        self.x += 6
        self.image(chart_path, link='', type='', w=700 / 4, h=450 / 4.9)

    def diff_tables(self, data, dataset):
        col_width = 45
        # self.set_xy(10.00125,40)
        x = self.x
        i = 0
        self.set_font("Times", 'B', size=10)
        line_height = self.font_size * 2.5
        self.x = x + 5
        self.multi_cell(90 * math.ceil(((len(data)) / 3)),
                        line_height,
                        'Accuracy Results' + ' (' + dataset + ')',
                        border=1,
                        align='C')
        y = self.y
        reset_y = self.y
        self.ln(line_height)
        self.set_font("Times", size=10)
        final_y = None
        for i, (key, val) in enumerate(data.items()):
            if i < 4:
                if (i % 3 == 0):
                    final_y = y
                    y = reset_y
                self.x = x + 90 * (i // 3) + 5
                self.y = y
                self.multi_cell(col_width,
                                line_height,
                                key,
                                border=1,
                                align='C')
                self.x += (45 + 90 * (i // 3)) + 5
                self.y = y
                self.multi_cell(col_width,
                                line_height,
                                str(val),
                                border=1,
                                align='C')
                y = self.y
                i += 1
        if final_y:
            self.y = final_y

    def graph_tables(self, data):
        real_data = []
        row_name = []
        row_data = []
        for key, val in data.items():
            row_name.append(key)
            row_data.append(str(val))
        real_data.append(row_name)
        real_data.append(row_data)

        col_width = 45
        self.set_xy(10.00125, 30)
        x = self.x
        self.x += 27
        self.set_font("Times", 'B', size=10)
        line_height = self.font_size * 2.5
        self.multi_cell(135,
                        line_height,
                        'Graph Compilation Results',
                        border=1,
                        align='C')
        y = self.y
        self.ln(line_height)
        self.set_font("Times", size=10)
        for row in real_data:
            self.x = x
            for i, datum in enumerate(row):
                self.y = y
                self.x += (i + 1) * 45 - 18
                self.multi_cell(col_width,
                                line_height,
                                str(datum),
                                border=1,
                                align='C')
            y = self.y
        self.y += 5

    def performance_tables(self, data):
        real_data = []
        row_name = []
        for i in range(len(data)):
            row_data = []
            for key, val in data[i].items():
                if i == 0:
                    row_name.append(key)
                row_data.append(val)
            real_data.append(row_data)
        real_data.insert(0, row_name)

        col_width = 33.75
        self.set_xy(10.00125, 65)
        x = self.x
        self.x += 27
        self.set_font("Times", 'B', size=10)
        line_height = self.font_size * 2.5
        self.multi_cell(135,
                        line_height,
                        'Performance Results',
                        border=1,
                        align='C')
        y = self.y
        self.ln(line_height)
        self.set_font("Times", size=10)
        for row in real_data:
            self.x = x
            for i, datum in enumerate(row):
                self.y = y
                self.x += (i + 1) * 33.75 - 6.75
                self.multi_cell(col_width,
                                line_height,
                                str(datum),
                                border=1,
                                align='C')
            y = self.y

            self.ln(line_height)

    def footer(self):
        # Go to 1.5 cm from bottom
        self.set_y(-15)
        # Select Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Print centered page number
        self.cell(0, 10, '%s' % self.page_no(), 0, 0, 'C')

    def generate_report(self, path):
        with open(path, 'r') as f:
            report = json.load(f)
        output_dir = os.path.dirname(path) + '/'
        index = output_dir.index('ByteMLPerf') + len('ByteMLPerf')
        base_path = output_dir[:index]

        icon_path = os.path.join(base_path, 'docs/images/icon.png')
        self.add_page()
        self.lines()
        self.icon(icon_path)
        self.graph_tables(report['Graph Compile'])
        if 'Performance' in report:
            self.performance_tables(report['Performance'])
        if 'Accuracy' in report:
            self.diff_tables(report['Accuracy'], report['Dataset'])
            if 'Diff Dist' in report['Accuracy']:
                self.charts(output_dir + report['Accuracy']['Diff Dist'])
        self.titles(report['Model'], report['Backend'])
        self.set_author('Bytedance')
        precision = path.split('/')[-1].split('-')[1]
        self.output(output_dir + report['Model'] + '-TO-' + precision.upper() + '.pdf', 'F')
        return True


def build_pdf(path):
    pdf = PDF(orientation='P', unit='mm', format='A4')
    return pdf.generate_report(path)
