class RandomForestResponseForm:
    @staticmethod
    def createForm(cm_before_smote, cm_after_smote, y_test, y_pred_before_smote, y_pred_smote, data):
        # 공통 정보 추출
        common_info = {
            'passengers_count': data[['num_passengers', 'booking_complete']].to_dict(orient='records'),
            'purchase_lead': data[['purchase_lead', 'booking_complete']].to_dict(orient='records'),
            'length_of_stay': data[['length_of_stay', 'booking_complete']].to_dict(orient='records'),
            'extra_baggage': data[['wants_extra_baggage', 'booking_complete']].to_dict(orient='records'),
            'preferred_seat': data[['wants_preferred_seat', 'booking_complete']].to_dict(orient='records'),
            'in_flight_meals': data[['wants_in_flight_meals', 'booking_complete']].to_dict(orient='records')
        }

        # 혼동 행렬 정보 구성
        cm_visualization_info_before_smote = {
            'cm': cm_before_smote.tolist(),
            'y_test': y_test.tolist(),
            'y_pred_before_smote': y_pred_before_smote.tolist()
        }

        cm_visualization_info_after_smote = {
            'cm': cm_after_smote.tolist(),
            'y_test': y_test.tolist(),
            'y_pred_smote': y_pred_smote.tolist()
        }

        return {
            'cm_visualization_info_before_smote': cm_visualization_info_before_smote,
            'cm_visualization_info_after_smote': cm_visualization_info_after_smote,
            'common_info': common_info
        }

    # def createForm(cm_before_smote, cm_after_smote, y_test, y_pred_before_smote, y_pred_smote, data):
    #     # 예약 완료 여부에 따른 승객 수 분포를 위한 정보
    #     passengers_count_info = {
    #         'x': data['num_passengers'].tolist(),  # Series를 리스트로 변환
    #         'hue': data['booking_complete'].tolist()  # Series를 리스트로 변환
    #     }
    #     print("pass passengers")
    #
    #     # 구매 리드 타임에 따른 예약 완료 여부를 위한 정보
    #     purchase_lead_info = {
    #         'x': data['purchase_lead'].tolist(),  # Series를 리스트로 변환
    #         'hue': data['booking_complete'].tolist()  # Series를 리스트로 변환
    #     }
    #     print("pass purchase_lead")
    #
    #     # 체류 기간에 따른 예약 완료 여부를 위한 정보
    #     length_of_stay_info = {
    #         'x': data['length_of_stay'].tolist(),  # Series를 리스트로 변환
    #         'hue': data['booking_complete'].tolist()  # Series를 리스트로 변환
    #     }
    #     print("pass length_of_stay_info")
    #
    #     # 추가 수하물 여부에 따른 예약 완료 여부를 위한 정보
    #     extra_baggage_info = {
    #         'x': data['wants_extra_baggage'].tolist(),  # Series를 리스트로 변환
    #         'hue': data['booking_complete'].tolist()  # Series를 리스트로 변환
    #     }
    #     print("pass extra_baggage_info")
    #
    #     # 선호 좌석 여부에 따른 예약 완료 여부를 위한 정보
    #     preferred_seat_info = {
    #         'x': data['wants_preferred_seat'].tolist(),  # Series를 리스트로 변환
    #         'hue': data['booking_complete'].tolist()  # Series를 리스트로 변환
    #     }
    #     print("pass preferred_seat_info")
    #
    #     # 기내식 여부에 따른 예약 완료 여부를 위한 정보
    #     in_flight_meals_info = {
    #         'x': data['wants_in_flight_meals'].tolist(),  # Series를 리스트로 변환
    #         'hue': data['booking_complete'].tolist()  # Series를 리스트로 변환
    #     }
    #     print("pass in_flight_meals_info")
    #
    #     cm_visualization_info_after_smote = {
    #         'cm': cm_after_smote.tolist(),  # ndarray를 리스트로 변환
    #         'y_test': y_test.tolist(),  # Series를 리스트로 변환
    #         'y_pred_smote': y_pred_smote.tolist()  # Series를 리스트로 변환
    #     }
    #     print("pass cm_visualization_info")
    #
    #     cm_visualization_info_before_smote = {
    #         'cm': cm_before_smote.tolist(),
    #         'y_test': y_test.tolist(),
    #         'y_pred_before_smote': y_pred_before_smote.tolist()
    #     }
    #
    #     return {
    #         'cm_visualization_info': cm_visualization_info_after_smote,
    #         'passengers_count_info': passengers_count_info,
    #         'purchase_lead_info': purchase_lead_info,
    #         'length_of_stay_info': length_of_stay_info,
    #         'extra_baggage_info': extra_baggage_info,
    #         'preferred_seat_info': preferred_seat_info,
    #         'in_flight_meals_info': in_flight_meals_info
    #     }
