#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import vaft
import argparse

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='Compare database and dumped raw signals')
    parser.add_argument('shot', type=int, help='Shot number to compare')
    parser.add_argument('--output', type=str, help='Output JSON file path (optional)')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum retries for DB connection')
    parser.add_argument('--daq-type', type=int, default=0, help='DAQ type')
    args = parser.parse_args()

    # DB 연결 초기화
    vaft.database.init_pool()

    # 1. JSON 파일로 저장
    print(f"Storing shot {args.shot} data as JSON...")
    vaft.database.raw.store_shot_as_json(
        shot=args.shot,
        output_path=args.output,
        plot_opt=1  # 시그널 플롯도 함께 저장
    )

    # 2. DB와 JSON 데이터 비교
    print("Comparing database and JSON data...")
    vaft.database.raw.compare_db_and_dumped_raw_signals(
        shot=args.shot,
        output_path=args.output,
        max_retries=args.max_retries,
        daq_type=args.daq_type
    )

if __name__ == "__main__":
    main() 