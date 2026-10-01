"""
Unit tests for DocumentClassifier.
Validates multi-signal detection across all KYC templates, unknown classification,
confidence thresholds, and non-reliance on filenames.
"""

import unittest
from kyc_sorter.detector.classifier import DocumentClassifier
from kyc_sorter.detector.templates import get_template_signatures


class TestDocumentClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = DocumentClassifier(confidence_threshold=0.80)

    def test_aadhaar_classification(self):
        text = """
        GOVERNMENT OF INDIA
        Unique Identification Authority of India
        To: John Doe
        DOB: 15/08/1990
        Male
        9876 5432 1098
        Help@uidai.gov.in 1947
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "AADHAAR_CARD")
        self.assertEqual(res.status, "IDENTIFIED")
        self.assertGreaterEqual(res.confidence, 0.80)
        self.assertIn("AADHAAR", res.matched_template)

    def test_pan_classification(self):
        text = """
        INCOME TAX DEPARTMENT
        GOVT. OF INDIA
        Permanent Account Number Card
        Name: RAJESH KUMAR
        Father's Name: SURESH KUMAR
        DOB: 12/05/1985
        ABCDE1234F
        Signature
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "PAN_CARD")
        self.assertEqual(res.status, "IDENTIFIED")
        self.assertGreaterEqual(res.confidence, 0.80)

    def test_voter_id_classification(self):
        text = """
        ELECTION COMMISSION OF INDIA
        ELECTOR PHOTO IDENTITY CARD
        EPIC No: WBF1234567
        Elector's Name: RAMESH CHANDRA
        Father's Name: DINESH CHANDRA
        Sex: Male
        Date of Birth: 01/01/1982
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "VOTER_ID")
        self.assertEqual(res.status, "IDENTIFIED")
        self.assertGreaterEqual(res.confidence, 0.80)

    def test_passport_classification(self):
        text = """
        PASSPORT
        REPUBLIC OF INDIA
        Passport No: Z1234567
        Surname: SHARMA
        Given Name: PRIYA
        Nationality: INDIAN
        Sex: F
        Date of Birth: 14/02/1995
        P<INDSHARMA<<PRIYA<<<<<<<<<<<<<<<<<<<<<<<<<<<
        Z1234567<8IND9502148F2802148<<<<<<<<<<<<<<<4
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "PASSPORT")
        self.assertEqual(res.status, "IDENTIFIED")
        self.assertGreaterEqual(res.confidence, 0.80)

    def test_driving_licence_classification(self):
        text = """
        UNION OF INDIA
        MOTOR VEHICLES DEPARTMENT
        DRIVING LICENCE
        DL Number: DL-0420110012345
        Authorisation to drive: LMV, MCWG
        Valid Till: 25-10-2035
        DOB: 10-10-1992
        Form 7 Rule 16
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "DRIVING_LICENCE")
        self.assertEqual(res.status, "IDENTIFIED")
        self.assertGreaterEqual(res.confidence, 0.80)

    def test_bank_statement_classification(self):
        text = """
        STATE BANK OF INDIA
        Savings Bank Passbook
        Account Number: 20123456789
        Customer ID: 85694213
        IFSC: SBIN0001234
        MICR: 110002034
        Branch: NEW DELHI MAIN
        Nomination Facility: Registered
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "BANK_STATEMENT")
        self.assertEqual(res.status, "IDENTIFIED")
        self.assertGreaterEqual(res.confidence, 0.80)

    def test_municipal_tax_receipt_classification(self):
        text = """
        MUNICIPAL CORPORATION OF GREATER MUMBAI
        ASSESSMENT & COLLECTION DEPARTMENT
        PROPERTY TAX RECEIPT
        Receipt No: MCR-2026-987654
        Ward No: W-12
        Property Tax Assessment Year: 2026-2027
        Amount Paid: Rs. 14,500.00
        Tax Paid in Full
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "MUNICIPAL_TAX_RECEIPT")
        self.assertEqual(res.status, "IDENTIFIED")
        self.assertGreaterEqual(res.confidence, 0.80)

    def test_govt_id_classification(self):
        text = """
        GOVERNMENT OF INDIA
        MINISTRY OF DEFENCE
        EMPLOYEE IDENTITY CARD / SERVICE IDENTITY CARD
        Employee No: GOV-887766
        Name: AJAY SINGH
        Designation: Under Secretary
        Official ID Card
        Signature of Issuing Authority
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "GOVT_ID")
        self.assertEqual(res.status, "IDENTIFIED")
        self.assertGreaterEqual(res.confidence, 0.80)

    def test_ckyc_declaration_classification(self):
        text = """
        CENTRAL KYC REGISTRY | CERSAI
        KYC DECLARATION FORM
        APPLICANT DECLARATION
        I hereby declare that the details furnished above are true and correct to the best of my knowledge.
        In case any of the above information is found to be false, I am aware that I may be held liable for it.
        Applicant Signature
        Date: 12/08/2026
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "CKYC_DECLARATION")
        self.assertEqual(res.status, "IDENTIFIED")
        self.assertGreaterEqual(res.confidence, 0.80)

    def test_multi_document_aadhaar_plus_pan(self):
        """A single image containing Aadhaar + PAN must route to UNKNOWN (Section 9)."""
        text = """
        GOVERNMENT OF INDIA
        Unique Identification Authority of India
        9876 5432 1098
        Help@uidai.gov.in 1947
        INCOME TAX DEPARTMENT
        GOVT. OF INDIA
        Permanent Account Number Card
        ABCDE1234F
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "UNKNOWN")
        self.assertEqual(res.status, "UNKNOWN")
        self.assertIn("Multiple KYC documents", res.matched_signals.get("reason", ""))

    def test_multi_document_pan_plus_voter(self):
        """A single image containing PAN + Voter ID must route to UNKNOWN (Section 9)."""
        text = """
        INCOME TAX DEPARTMENT
        Permanent Account Number
        ABCDE1234F
        ELECTION COMMISSION OF INDIA
        ELECTOR PHOTO IDENTITY CARD
        EPIC No: WBF1234567
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "UNKNOWN")
        self.assertEqual(res.status, "UNKNOWN")
        self.assertIn("Multiple KYC documents", res.matched_signals.get("reason", ""))

    def test_multi_document_aadhaar_plus_passport(self):
        """A single image containing Aadhaar + Passport must route to UNKNOWN (Section 9)."""
        text = """
        Unique Identification Authority of India
        9876 5432 1098
        REPUBLIC OF INDIA
        PASSPORT
        Passport No: Z1234567
        P<INDSHARMA<<PRIYA<<<<<<<<<<<<<<<<<<<<<<<<<<<
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "UNKNOWN")
        self.assertEqual(res.status, "UNKNOWN")
        self.assertIn("Multiple KYC documents", res.matched_signals.get("reason", ""))

    def test_multi_document_pan_plus_driving_licence(self):
        """A single image containing PAN + Driving Licence must route to UNKNOWN (Section 9)."""
        text = """
        INCOME TAX DEPARTMENT
        Permanent Account Number Card
        ABCDE1234F
        UNION OF INDIA
        MOTOR VEHICLES DEPARTMENT
        DRIVING LICENCE
        DL Number: DL-0420110012345
        Authorisation to drive: LMV
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "UNKNOWN")
        self.assertEqual(res.status, "UNKNOWN")
        self.assertIn("Multiple KYC documents", res.matched_signals.get("reason", ""))

    def test_multi_document_aadhaar_plus_pan_plus_voter(self):
        """A single image containing Aadhaar + PAN + Voter ID must route to UNKNOWN (Section 9)."""
        text = """
        Unique Identification Authority of India
        9876 5432 1098
        INCOME TAX DEPARTMENT
        Permanent Account Number
        ABCDE1234F
        ELECTION COMMISSION OF INDIA
        ELECTOR PHOTO IDENTITY CARD
        EPIC No: WBF1234567
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "UNKNOWN")
        self.assertEqual(res.status, "UNKNOWN")
        self.assertIn("Multiple KYC documents", res.matched_signals.get("reason", ""))

    def test_electricity_bill_classification(self):
        text = """
        TNEB POWER DISTRIBUTION
        ELECTRICITY BILL
        Consumer No: 04-123-456-789
        Meter No: M123456
        Billing Period: 01-AUG-2026 TO 31-AUG-2026
        Units Consumed: 340 KWH
        Energy Charges: 1540.00
        Total Amount Payable: 1720.00
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "ELECTRICITY_BILL")
        self.assertEqual(res.status, "IDENTIFIED")

    def test_gas_bill_classification(self):
        text = """
        BHARAT PETROLEUM CORPORATION
        BHARATGAS LPG CONNECTION CARD
        Card Serial No: WR-9876543
        Consumer No: 12345678
        Type of Cylinders: 14.2 KG Domestic
        LPGID: 10000000123456789
        Distributor Code: 112233
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "GAS_BILL")
        self.assertEqual(res.status, "IDENTIFIED")

    def test_ration_card_classification(self):
        text = """
        FOOD AND CIVIL SUPPLIES DEPARTMENT
        HOUSEHOLD CARD / RATION CARD
        Ration Card Number: WAP123456789
        Name of Head of Household: SITA DEVI
        Fair Price Shop No: 42
        Family Members Details: 4
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "RATION_CARD")
        self.assertEqual(res.status, "IDENTIFIED")

    def test_phone_bill_classification(self):
        text = """
        BHARAT SANCHAR NIGAM LTD (BSNL)
        YOUR MONTHLY BILL FOR MOBILE SERVICES
        Mobile Number: 9876543210
        Bill Number: BSNL-2026-998877
        Bill Period: 01/08/2026 to 31/08/2026
        Amount Payable: Rs. 649.00
        Payment Due Date: 15/09/2026
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "PHONE_BILL")
        self.assertEqual(res.status, "IDENTIFIED")

    def test_ckyc_confirmation_classification(self):
        text = """
        CENTRAL KYC REGISTRY | CERSAI
        KNOW YOUR CUSTOMER (KYC) APPLICATION FORM
        CUSTOMER INFORMATION FORM
        KYC Number: 10023456789012
        Proof of Identity: PAN Card
        Proof of Address: Passport
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "CKYC_CONFIRMATION")
        self.assertEqual(res.status, "IDENTIFIED")

    def test_registration_certificate_classification(self):
        text = """
        TRANSPORT DEPARTMENT
        CERTIFICATE OF REGISTRATION
        Regn No: DL01AB1234
        Chassis Number: MA3ERB12S00123456
        Engine Number: K12M1234567
        Vehicle Class: Motor Car (LMV)
        Seating Capacity: 5
        Registering Authority: Delhi RTO
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "REGISTRATION_CERTIFICATE")
        self.assertEqual(res.status, "IDENTIFIED")

    def test_it_returns_classification(self):
        text = """
        INDIAN INCOME TAX RETURN ACKNOWLEDGEMENT
        Assessment Year: 2026-27
        Form: ITR-1 SAHAJ
        Gross Total Income: 8,50,000
        Total Tax and Interest Payable: 45,000
        E-Filing Acknowledgement Number: 123456789012345
        Verified Electronically
        """
        res = self.classifier.classify_text(text)
        self.assertEqual(res.document_type, "IT_RETURNS")
        self.assertEqual(res.status, "IDENTIFIED")

    def test_non_kyc_goes_to_unknown(self):
        random_texts = [
            "Today special dish: Butter Chicken, Garlic Naan, Paneer Tikka. Table 5 order.",
            "Dear John, Hope you are having a wonderful vacation. Regards, Sarah.",
            "Meeting notes: Sprint planning agenda for Q4. Team velocity and retrospectives.",
            "",
            "   \n\t  ",
        ]
        for t in random_texts:
            res = self.classifier.classify_text(t)
            self.assertEqual(res.document_type, "UNKNOWN")
            self.assertEqual(res.status, "UNKNOWN")

    def test_low_confidence_threshold_routes_to_unknown(self):
        # A single generic keyword without strong corroboration scores below high threshold
        strict_classifier = DocumentClassifier(confidence_threshold=0.95)
        weak_text = "Name: John Doe, Father: Bob Doe, Male, DOB: 01/01/1990"
        res = strict_classifier.classify_text(weak_text)
        self.assertEqual(res.document_type, "UNKNOWN")
        self.assertEqual(res.status, "UNKNOWN")

    def test_space_collapsed_ocr_variant(self):
        # PaddleOCR sometimes collapses spaces on low-res scans
        garbled_text = """
        uniqueidentificationauthorityofindia
        9876 5432 1098
        year of birth: 1988
        """
        res = self.classifier.classify_text(garbled_text)
        self.assertEqual(res.document_type, "AADHAAR_CARD")
        self.assertEqual(res.status, "IDENTIFIED")

    def test_filename_is_not_used_for_classification(self):
        # Document contains clear PAN text; classifier function only takes OCR text
        # and has no awareness of any fake filename like 'aadhaar.jpg'
        pan_text = """
        INCOME TAX DEPARTMENT
        Permanent Account Number
        ABCDE1234F
        """
        res = self.classifier.classify_text(pan_text)
        self.assertEqual(res.document_type, "PAN_CARD")


if __name__ == "__main__":
    unittest.main()
