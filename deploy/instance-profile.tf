resource "aws_iam_role" "dl_iam_role" {
  name               = "dl_iam_role"
  assume_role_policy = "${data.aws_iam_policy_document.assume_role_policy.json}"
}

resource "aws_iam_policy" "dl_iam_policy" {
  name        = "dl_iam_policy"
  policy      = "${data.aws_iam_policy_document.s3_bucket_policy.json}"
}

resource "aws_iam_policy_attachment" "dl_iam_attach" {
  name       = "dl_iam_attachment"
  roles      = ["${aws_iam_role.dl_iam_role.name}"]
  policy_arn = "${aws_iam_policy.dl_iam_policy.arn}"
}

resource "aws_iam_instance_profile" "dl_iam_instance_profile" {
  name  = "dl_iam_instance_profile"
  role = "${aws_iam_role.dl_iam_role.name}"
}

data "aws_iam_policy_document" "assume_role_policy" {
  statement {
    actions = [ "sts:AssumeRole" ]

    principals {
      type        = "Service"
      identifiers = [ "ec2.amazonaws.com" ]
    }
  }
}

data "aws_iam_policy_document" "s3_bucket_policy" {
  statement {
    actions     = [ "s3:ListBucket" ]
    resources   = [ "arn:aws:s3:::${var.dataset_bucket}" ]
  }

  statement {
    actions     = [ "s3:GetObject", "s3:PutObject" ]
    resources   = [ "arn:aws:s3:::${var.dataset_bucket}/*" ]
  }
}